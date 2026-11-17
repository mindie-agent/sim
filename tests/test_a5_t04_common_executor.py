"""Behavioral contract for the shared fragment executor and bounded advance.

These are synthetic logical fixtures. The 4/5/6 us example is scaled 1:1 onto
the Engine's millisecond clock and labeled a fixture, not a hardware claim.
No test loads weights, initializes a device or consumes a target duration.
"""
import pytest

from mindie_sim.prediction.engine import (
    Allocation, Engine, ExecutionError, Resource,
)
from mindie_sim.prediction.scheduling import (
    FragmentConductor, FragmentContractError, UnsupportedFragmentSemantics,
)
from mindie_sim.schema.invocation import Occurrence
from mindie_sim.schema.operators import (
    AllocationEffect, ProgramFragment, ResourcePolicy, StateEffect, WorkItem,
    WorkSet,
)
from mindie_sim.schema.subgraphs import EventPortSpec, PortSet


def _fragment(name, work=(), *, repeats=1, allocations=(), state_effects=(),
              ports=None, occurrence=None, coupled=False, priority=0,
              policies=()):
    items = tuple(WorkItem(resource, amount, unit)
                  for resource, amount, unit in work)
    return ProgramFragment(
        name, work=WorkSet(items), repeats=repeats, allocations=allocations,
        state_effects=state_effects, ports=ports or PortSet(),
        occurrence=occurrence, coupled=coupled, priority=priority,
        policies=policies)


def test_sharing_policy_and_priority_change_successor_completion_only():
    expected = {
        ("fifo", 0, 0): (1, 3, 4),
        ("fair", 0, 0): (2, 3, 5),
        ("fifo", 1, 0): (3, 2, 6),
    }
    for (policy, priority_a, priority_b), times in expected.items():
        engine = Engine([
            Resource("H", 64, unit="bytes", policy=policy),
            Resource("C", 1, unit="compute_service_us", policy="fifo"),
        ])
        a = engine.submit("A", {"H": 64}, priority=priority_a)
        b = engine.submit("B", {"H": 128}, priority=priority_b)
        c = engine.submit("C_after_A", {"C": 3}, after=(a,))
        engine.advance(until=(c,), first=False)
        assert (a.time, b.time, c.time) == times
        assert engine.serviced == {"H": 192.0, "C": 3.0}
        assert engine.summary()["live_tasks"] == 0


def test_fifo_priority_is_admission_order_not_preemption():
    engine = Engine([Resource("H", 64, unit="bytes", policy="fifo")])
    first = engine.submit("first", {"H": 128}, priority=1)
    engine.advance(horizon=1.0)
    late = engine.submit("late", {"H": 64}, priority=0)
    engine.advance(until=(first, late), first=False)
    # The already-started lower-priority task keeps service; the higher
    # priority arrival is admitted later but does not displace it.
    assert first.time == 2
    assert late.time == 3
    assert engine.serviced == {"H": 192.0}


def test_non_preemptive_slot_is_held_without_inventing_work():
    engine = Engine([Resource("C", 64, unit="credits")], pools={"S": 1})
    conductor = FragmentConductor(engine)
    external_release = engine.submit("external_release", latency=5)
    a = conductor.submit(
        _fragment("A", (("C", 128, "credits"),),
                  allocations=(AllocationEffect.acquire("A.slot", "S", 1),)),
        holds={"A.slot": external_release})
    arrival = engine.submit("B_arrival", latency=1)
    b = conductor.submit(
        _fragment("B", (("C", 64, "credits"),),
                  allocations=(AllocationEffect.acquire("B.slot", "S", 1),
                               AllocationEffect.release("B.slot", "S", 1))),
        after=(arrival,))

    conductor.advance(horizon=4.0)
    assert a.event.time == 2
    assert any(name.endswith(":A.slot") for name in engine.allocations)
    assert b.event.started is None

    conductor.advance()
    assert a.release_events[0].time == 5
    assert a.release_events[0].time - a.event.started == 5
    assert b.event.started == 5
    assert b.event.time == 6
    assert engine.serviced == {"C": 192.0}
    assert engine.busy_ms == {"C": 3.0}
    assert not engine.allocations


def test_coupled_progress_keeps_both_resources_busy_without_extra_service():
    coupled_engine = Engine([Resource("r1", 10, unit="u"),
                             Resource("r2", 10, unit="u")])
    coupled = FragmentConductor(coupled_engine)
    done = coupled.submit(_fragment("steady", (("r1", 30, "u"), ("r2", 10, "u")),
                                    coupled=True))
    coupled.advance()
    assert done.event.time == 3
    assert coupled_engine.serviced == {"r1": 30.0, "r2": 10.0}
    assert coupled_engine.busy_ms == {"r1": 3.0, "r2": 3.0}

    separate_engine = Engine([Resource("r1", 10, unit="u"),
                              Resource("r2", 10, unit="u")])
    separate = FragmentConductor(separate_engine)
    plain = separate.submit(_fragment("plain", (("r1", 30, "u"), ("r2", 10, "u"))))
    separate.advance()
    assert plain.event.time == 3
    assert separate_engine.serviced == {"r1": 30.0, "r2": 10.0}
    # r2 finishes after its own work, so it is not retained as "busy".
    assert separate_engine.busy_ms == {"r1": 3.0, "r2": 1.0}


def test_homogeneous_repeats_batch_and_match_an_expanded_reference():
    batched_engine = Engine([Resource("hbm", 64, unit="bytes")])
    batched = FragmentConductor(batched_engine)
    handle = batched.submit(_fragment("transfer", (("hbm", 64, "bytes"),),
                                      repeats=3))
    batched.advance()
    assert handle.batched
    assert handle.repeat_count == 3
    assert handle.event.time == 3
    assert batched_engine.serviced == {"hbm": 192.0}
    assert batched_engine.completed_count == 1
    assert batched_engine.summary()["peak_live_tasks"] <= 2

    expanded_engine = Engine([Resource("hbm", 64, unit="bytes")])
    expanded = FragmentConductor(expanded_engine)
    previous = None
    events = []
    for index in range(3):
        handle = expanded.submit(
            _fragment(f"tile{index}", (("hbm", 64, "bytes"),)),
            after=(() if previous is None else (previous,)))
        previous = handle.event
        events.append(handle.event)
    expanded.advance()
    assert [event.time for event in events] == [1, 2, 3]
    assert expanded_engine.serviced == {"hbm": 192.0}
    assert expanded_engine.completed_count == 3

    with pytest.raises(FragmentContractError,
                       match="fragment_batched_work_overflow"):
        batched.submit(_fragment("huge", (("hbm", 1, "bytes"),),
                                 repeats=10 ** 400))


def test_repeats_with_allocation_expand_bounded_and_release_the_slot():
    engine = Engine([Resource("hbm", 64, unit="bytes")], pools={"L2": 64})
    conductor = FragmentConductor(engine)
    handle = conductor.submit(_fragment(
        "staged", (("hbm", 64, "bytes"),), repeats=2,
        allocations=(AllocationEffect.acquire("staging", "L2", 64),
                     AllocationEffect.release("staging", "L2", 64))))
    conductor.advance()
    assert not handle.batched
    assert handle.event.time == 2
    assert engine.serviced == {"hbm": 128.0}
    assert engine.peak_allocated == {"L2": 64.0}
    assert engine.allocated == {"L2": 0.0}
    assert engine.completed_count == 2

    with pytest.raises(UnsupportedFragmentSemantics,
                       match="fragment_repeats_exceed_expansion_bound"):
        FragmentConductor(engine, repeat_limit=1).submit(_fragment(
            "too_many", (("hbm", 64, "bytes"),), repeats=2,
            allocations=(AllocationEffect.acquire("staging", "L2", 64),
                         AllocationEffect.release("staging", "L2", 64))))
    # The bound is checked before a program grant can be submitted, so a
    # rejected fragment leaves no live allocation or task behind.
    clean = Engine([Resource("hbm", 64, unit="bytes")], pools={"L2": 64})
    with pytest.raises(UnsupportedFragmentSemantics,
                       match="fragment_repeats_exceed_expansion_bound"):
        FragmentConductor(clean, repeat_limit=1).submit(_fragment(
            "too_many_program", (("hbm", 64, "bytes"),), repeats=2,
            allocations=(AllocationEffect.acquire(
                "staging", "L2", 64, lifetime="program"),
                AllocationEffect.release(
                    "staging", "L2", 64, lifetime="program"))))
    assert clean.summary()["live_tasks"] == 0
    assert clean.allocated == {"L2": 0.0}


def test_conditional_allocation_is_not_silently_made_unconditional():
    engine = Engine([Resource("hbm", 64, unit="bytes")], pools={"L2": 128})
    conductor = FragmentConductor(engine)
    fragment = _fragment(
        "conditional", (("hbm", 64, "bytes"),),
        allocations=(AllocationEffect.acquire("buf", "L2", 128,
                                              condition="use_buf"),
                     AllocationEffect.release("buf", "L2", 128,
                                              condition="use_buf")))
    with pytest.raises(FragmentContractError,
                       match="conditional_effect_unbound:buf:use_buf"):
        conductor.submit(fragment)

    conductor.submit(fragment, conditions={"use_buf": False})
    conductor.advance()
    assert engine.peak_allocated == {"L2": 0.0}

    conductor.submit(fragment, conditions={"use_buf": True})
    conductor.advance()
    assert engine.peak_allocated == {"L2": 128.0}
    assert engine.allocated == {"L2": 0.0}


def test_program_scoped_allocation_spans_repeats_and_event_scope_is_explicit():
    engine = Engine([Resource("hbm", 64, unit="bytes")], pools={"L2": 64})
    conductor = FragmentConductor(engine)
    handle = conductor.submit(_fragment(
        "resident", (("hbm", 64, "bytes"),), repeats=2,
        allocations=(AllocationEffect.acquire("buf", "L2", 64,
                                              lifetime="program"),
                     AllocationEffect.release("buf", "L2", 64,
                                              lifetime="program"))))
    conductor.advance(horizon=1.0)
    assert engine.allocations  # still resident while repeat 2 is pending
    conductor.advance()
    assert handle.event.time == 2
    assert engine.peak_allocated == {"L2": 64.0}
    assert engine.allocated == {"L2": 0.0}

    event_scoped = _fragment(
        "event_scope", (("hbm", 64, "bytes"),),
        allocations=(AllocationEffect.acquire("buf", "L2", 64,
                                              lifetime="event"),))
    with pytest.raises(UnsupportedFragmentSemantics,
                       match="allocation_scope_unsupported:event"):
        conductor.submit(event_scoped)


def test_externally_held_program_allocation_releases_on_its_event():
    engine = Engine([Resource("hbm", 64, unit="bytes")], pools={"L2": 64})
    conductor = FragmentConductor(engine)
    release = engine.submit("external_release", latency=5)
    handle = conductor.submit(_fragment(
        "held", (("hbm", 64, "bytes"),),
        allocations=(AllocationEffect.acquire("buf", "L2", 64,
                                              lifetime="program"),)),
        holds={"buf": release})
    conductor.advance()
    assert handle.event.time == 1
    conductor.advance(until=(handle.release_events[0],))
    assert handle.release_events[0].time == 5
    assert engine.allocated == {"L2": 0.0}


def test_state_visibility_keeps_rejected_candidate_out_of_committed_state():
    engine = Engine(())
    conductor = FragmentConductor(engine, state={"kv": 7})
    draft = conductor.submit(_fragment(
        "draft", state_effects=(StateEffect("kv", "read", version=7),
                                StateEffect("kv", "write", version=8))))
    conductor.submit(_fragment(
        "reject", state_effects=(StateEffect("kv", "rollback", version=8),)),
        after=(draft.event,))
    conductor.advance()
    assert conductor.state.snapshot() == {
        "committed": {"kv": 7}, "live": {"kv": (7,)}}

    accepting = FragmentConductor(Engine(()), state={"kv": 7})
    draft = accepting.submit(_fragment(
        "draft", state_effects=(StateEffect("kv", "read", version=7),
                                StateEffect("kv", "write", version=8))))
    accepting.submit(_fragment(
        "accept", state_effects=(StateEffect("kv", "commit", version=8),)),
        after=(draft.event,))
    accepting.advance()
    assert accepting.state.committed_version("kv") == 8


def test_state_read_against_unavailable_version_fails_explicitly():
    conductor = FragmentConductor(Engine(()), state={"kv": 7})
    conductor.submit(_fragment(
        "stale", state_effects=(StateEffect("kv", "read", version=9),)))
    with pytest.raises(FragmentContractError, match="state_version_unavailable"):
        conductor.advance()


def test_conditional_state_effect_and_repeated_reads_are_bound_at_their_events():
    accepting = FragmentConductor(Engine(()), state={"kv": 7})
    write = accepting.submit(_fragment(
        "write", state_effects=(StateEffect("kv", "write", version=8),)))
    accepting.submit(_fragment(
        "no_commit", state_effects=(StateEffect(
            "kv", "commit", version=8, condition="commit_now"),)),
        after=(write.event,), conditions={"commit_now": False})
    accepting.advance()
    assert accepting.state.committed_version("kv") == 7

    repeated = FragmentConductor(
        Engine([Resource("hbm", 64, unit="bytes")]), state={"kv": 7})
    handle = repeated.submit(_fragment(
        "read_twice", (("hbm", 64, "bytes"),), repeats=2,
        state_effects=(StateEffect("kv", "read", version=7),)))
    repeated.advance()
    assert not handle.batched
    assert handle.event.time == 2
    assert repeated.state.committed_version("kv") == 7


def test_committed_state_is_visible_to_a_same_timestamp_successor():
    conductor = FragmentConductor(
        Engine([Resource("x", 1, unit="u")]), state={"kv": 7})
    write = conductor.submit(_fragment(
        "write", state_effects=(StateEffect("kv", "write", version=8),)))
    commit = conductor.submit(_fragment(
        "commit", state_effects=(StateEffect("kv", "commit", version=8),)),
        after=(write.event,))
    reader = conductor.submit(_fragment(
        "reader", (("x", 1, "u"),),
        state_effects=(StateEffect("kv", "read", version=8),)),
        after=(commit.event,))
    conductor.advance()
    assert reader.event.started == commit.event.time == 0
    assert reader.event.time == 1
    assert conductor.state.committed_version("kv") == 8


def test_state_ledger_reclaims_superseded_versions_in_a_long_commit_chain():
    conductor = FragmentConductor(
        Engine([Resource("step", 1, unit="u")]), state={"kv": 0})
    for version in range(1, 21):
        write = conductor.submit(_fragment(
            "write", (("step", 1, "u"),),
            state_effects=(StateEffect("kv", "write", version=version),)))
        commit = conductor.submit(_fragment(
            "commit", state_effects=(StateEffect("kv", "commit",
                                                 version=version),)),
            after=(write.event,))
        conductor.advance(until=(commit.event,))
    assert conductor.state.committed_version("kv") == 20
    assert len(conductor.state.live_versions("kv")) <= 2


def test_same_timestamp_state_commits_require_an_explicit_arbiter():
    def build(arbiters, reverse=False):
        engine = Engine(())
        conductor = FragmentConductor(engine, state={"kv": 7})
        w8 = conductor.submit(_fragment(
            "w8", state_effects=(StateEffect("kv", "write", version=8),)))
        w9 = conductor.submit(_fragment(
            "w9", state_effects=(StateEffect("kv", "write", version=9),)))
        waits = (w8.event, w9.event)
        commit8 = StateEffect("kv", "commit", version=8, arbiter=arbiters[0])
        commit9 = StateEffect("kv", "commit", version=9, arbiter=arbiters[1])
        order = ((commit8, "c8"), (commit9, "c9"))
        if reverse:
            order = tuple(reversed(order))
        for effect, name in order:
            conductor.submit(_fragment(name, state_effects=(effect,)), after=waits)
        return conductor

    with pytest.raises(FragmentContractError,
                       match="state_effect_conflict_requires_arbiter"):
        build(("", "")).advance()
    # The arbiter is a total order key, not container iteration order: the
    # same two commits select the same version when their arbiters are swapped
    # or when the submissions are reversed.
    for arbiters, winner in ((("a", "b"), 9), (("b", "a"), 8)):
        for reverse in (False, True):
            conductor = build(arbiters, reverse=reverse)
            conductor.advance()
            assert conductor.state.committed_version("kv") == winner


def test_stream_order_is_explicit_and_event_ports_resolve_the_recorded_event():
    engine = Engine([Resource("hbm", 64, unit="bytes")])
    conductor = FragmentConductor(engine)
    producer = conductor.submit(_fragment(
        "producer", (("hbm", 64, "bytes"),),
        occurrence=Occurrence("stage", "producer", stream="s"),
        ports=PortSet(event_out=(EventPortSpec("done", "out", "complete"),))))
    consumer = conductor.submit(_fragment(
        "consumer", (("hbm", 64, "bytes"),),
        occurrence=Occurrence("stage", "consumer", stream="s"),
        ports=PortSet(event_in=(EventPortSpec("done", "in", "ready"),))))
    independent = conductor.submit(_fragment(
        "independent", (("hbm", 64, "bytes"),),
        occurrence=Occurrence("stage", "independent", stream="other")))
    conductor.advance()
    assert consumer.event.started == producer.event.time == 2
    assert consumer.event.time == 3
    assert independent.event.started == 0
    assert independent.event.time == 2
    assert engine.serviced == {"hbm": 192.0}


def test_missing_required_event_and_optional_event_are_distinguished():
    conductor = FragmentConductor(Engine([Resource("hbm", 1, unit="u")]))
    required = _fragment(
        "required", (("hbm", 1, "u"),),
        ports=PortSet(event_in=(EventPortSpec("ready", "in", "ready"),)))
    with pytest.raises(FragmentContractError, match="fragment_input_unbound:ready"):
        conductor.submit(required)

    optional = _fragment(
        "optional", (("hbm", 1, "u"),),
        ports=PortSet(event_in=(EventPortSpec(
            "ready", "in", "ready", optional=True),)))
    handle = conductor.submit(optional)
    conductor.advance()
    assert handle.event.time == 1


def test_long_interactive_sequence_keeps_only_active_and_bounded_caches():
    engine = Engine([Resource("wire", 1, unit="u")])
    conductor = FragmentConductor(engine, cache_limit=8)
    peak_live = 0
    for index in range(400):
        handle = conductor.submit(_fragment(
            f"op{index}", (("wire", 1, "u"),), repeats=100))
        conductor.advance(until=(handle.event,))
        peak_live = max(peak_live, engine.summary()["peak_live_tasks"])
    summary = conductor.summary()
    assert summary["live_fragments"] == 0
    assert summary["engine"]["live_tasks"] == 0
    assert summary["totals"]["fragments"] == 400
    assert summary["totals"]["repeats"] == 40_000
    assert summary["totals"]["batched"] == 400
    assert len(summary["fragments"]) <= 8
    assert summary["aggregates_evicted"] > 0
    assert summary["recent"] == []
    assert peak_live <= 2


def test_default_and_detailed_observations_share_the_same_execution():
    def run(detail):
        engine = Engine([Resource("hbm", 64, unit="bytes")], trace_limit=4)
        conductor = FragmentConductor(engine, detail=detail, cache_limit=4)
        events = []
        for index in range(6):
            handle = conductor.submit(
                _fragment(f"op{index}", (("hbm", 64, "bytes"),)))
            events.append(handle.event)
        conductor.advance()
        return (conductor, engine, events)

    default, default_engine, default_events = run(False)
    detailed, detailed_engine, detailed_events = run(True)
    assert [event.time for event in default_events] == \
        [event.time for event in detailed_events]
    assert default_engine.summary() == detailed_engine.summary()
    assert default.summary()["totals"] == detailed.summary()["totals"]
    assert default.summary()["recent"] == []
    assert len(detailed.summary()["recent"]) == 4
    assert detailed.summary()["recent_dropped"] == 2
    assert detailed.summary()["aggregates_evicted"] == 2


def test_step_budget_and_zero_progress_chains_terminate_explicitly():
    engine = Engine([Resource("x", 1)])
    previous = None
    events = []
    for index in range(4):
        previous = engine.submit(
            f"step{index}", {"x": 1},
            after=(() if previous is None else (previous,)))
        events.append(previous)
    with pytest.raises(ExecutionError, match="execution_step_budget_exhausted"):
        engine.advance(until=(events[-1],), max_steps=2)
    # The budget is per advance call, not a lifetime cap on useful progress.
    engine.advance(until=(events[-1],), max_steps=2)
    assert events[-1].time == 4

    spinning = Engine([Resource("x", 1)], zero_progress_limit=6)

    def again(event):
        spinning.submit("again", after=(event,), on_complete=again)

    spinning.submit("start", on_complete=again)
    with pytest.raises(ExecutionError, match="execution_zero_progress"):
        spinning.advance()


def test_concurrent_instances_of_one_fragment_do_not_share_allocation_names():
    engine = Engine([Resource("hbm", 64, unit="bytes")], pools={"L2": 128})
    conductor = FragmentConductor(engine)
    fragment = _fragment(
        "same", (("hbm", 64, "bytes"),),
        allocations=(AllocationEffect.acquire("buf", "L2", 64),
                     AllocationEffect.release("buf", "L2", 64)))
    first = conductor.submit(fragment)
    second = conductor.submit(fragment)
    conductor.advance()
    assert first.event.time == second.event.time == 2
    assert engine.peak_allocated == {"L2": 128.0}
    assert engine.allocated == {"L2": 0.0}


def test_event_registry_reclaims_old_records_and_reports_the_miss():
    engine = Engine([Resource("x", 1, unit="u")])
    conductor = FragmentConductor(engine, event_limit=2)
    for index in range(4):
        conductor.record_event("done", engine.submit(f"e{index}", latency=1))
    assert conductor.summary()["recorded_events"] == 2
    assert conductor.summary()["events_evicted"] == 2
    stale = _fragment(
        "stale", (("x", 1, "u"),),
        ports=PortSet(event_in=(EventPortSpec(
            "done", "in", "ready", generation=0),)))
    with pytest.raises(FragmentContractError, match="fragment_input_unbound:done"):
        conductor.submit(stale)
    implicit = _fragment(
        "implicit", (("x", 1, "u"),),
        ports=PortSet(event_in=(EventPortSpec("done", "in", "ready"),)))
    with pytest.raises(FragmentContractError,
                       match="fragment_event_generation_required:done"):
        conductor.submit(implicit)


def test_ambiguous_event_generation_requires_an_explicit_generation():
    engine = Engine([Resource("x", 1, unit="u")])
    conductor = FragmentConductor(engine)
    first = engine.submit("first", latency=1)
    second = engine.submit("second", latency=2)
    conductor.record_event("done", first)
    conductor.record_event("done", second)
    ambiguous = _fragment(
        "ambiguous", (("x", 1, "u"),),
        ports=PortSet(event_in=(EventPortSpec("done", "in", "ready"),)))
    with pytest.raises(FragmentContractError,
                       match="fragment_event_generation_required:done"):
        conductor.submit(ambiguous)

    bound = _fragment(
        "bound", (("x", 1, "u"),),
        ports=PortSet(event_in=(EventPortSpec(
            "done", "in", "ready", generation=1),)))
    handle = conductor.submit(bound)
    conductor.advance()
    assert handle.event.started == 2
    assert handle.event.time == 3


def test_engine_boundaries_distinguish_start_from_completion():
    engine = Engine([Resource("host", 1, policy="fifo"), Resource("x", 1)])
    prepare = engine.submit("prepare", {"host": 2})
    launch = engine.submit("launch", {"host": 1}, after=(prepare,))
    started = engine.start_boundary(launch, "launch_started")
    prepared = engine.boundary("prepared", after=(prepare,))
    engine.advance(until=(launch,))
    assert started.time == started.started == prepared.time == launch.started == 2
    assert launch.time == 3
    late = engine.start_boundary(launch, "late_start_marker")
    assert late.time == 2

    lazy = engine.future("lazy_program")
    with pytest.raises(ExecutionError, match="start_boundary_requires_active_task"):
        engine.start_boundary(lazy)


def test_release_without_a_live_grant_is_not_silently_ignored():
    engine = Engine((), pools={"L2": 16})
    engine.submit("free", free=(Allocation("missing", "L2", 16),))
    with pytest.raises(ExecutionError, match="release_without_live_allocation"):
        engine.advance()


def test_execution_step_host_submit_events_are_real_and_absent_without_launch():
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import IterationWork, OpSpec
    from mindie_sim.schema.workload import IterationPlan, RequestView
    from test_prediction_system import case

    parsed = parse_case(case())
    execution = Execution(parsed, CostModel(parsed))
    plan = IterationPlan((RequestView("r", 1, 4, True, "decode"),))
    operator = OpSpec(
        "copy", "test", "prefill", 0, 0, (), (1,), (1,), (1,), (1,),
        hbm_bytes=1_000_000, dtype="bf16")
    step = execution.add(
        IterationWork((operator,), plan, 1, 1, 1, None, "body", "test"), 0, 0)
    assert len(step.host_submits) == 1
    start = execution.stage_submit_boundary(0, 0, step=step)
    assert start is not None
    execution.engine.advance(until=(step.completion,), first=False)
    assert start.time == step.host_submits[0][2].time == 0.0

    empty = execution.add(
        IterationWork((), plan, 2, 0, 0, None, "body", "empty"), 0, 0)
    assert empty.host_submits == ()
    assert execution.stage_submit_boundary(0, 0, step=empty) is None
    execution.engine.advance(until=(empty.completion,), first=False)


def test_operator_step_and_fragment_share_one_engine_context():
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import IterationWork, OpSpec
    from mindie_sim.schema.workload import IterationPlan, RequestView
    from test_prediction_system import case

    parsed = parse_case(case())
    execution = Execution(parsed, CostModel(parsed))
    plan = IterationPlan((RequestView("r", 1, 4, True, "decode"),))
    operator = OpSpec(
        "copy", "test", "prefill", 0, 0, (), (1,), (1,), (1,), (1,),
        hbm_bytes=1_000_000, dtype="bf16")
    step = execution.add(
        IterationWork((operator,), plan, 1, 1, 1, None, "body", "test"), 0, 0)
    conductor = execution.fragment_conductor(replica=0, stage=0)
    follow = conductor.submit(
        _fragment("follow", (("hbm", 1_000_000, "bytes"),)),
        after=(step.device_complete,))
    execution.engine.advance(until=(follow.event,), first=False)
    assert follow.event.started >= step.device_complete.time
    assert execution.engine.summary()["live_tasks"] == 0
    assert execution.engine.serviced["memory@r0:s0"] == 2_000_000


def test_execution_entry_binds_fragments_to_the_shared_stage_engine():
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    from test_prediction_system import case

    parsed = parse_case(case())
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    assert execution.stage_submit_boundary(replica=0, stage=0) is None
    conductor = execution.fragment_conductor(replica=0, stage=0)
    handle = conductor.submit(
        _fragment("probe", (("hbm", 1_000_000, "bytes"),)))
    conductor.advance(until=(handle.event,))
    assert handle.event.time > 0
    assert execution.engine.serviced["memory@r0:s0"] == 1_000_000

    cached = execution.submit_fragment(
        _fragment("probe2", (("hbm", 1_000_000, "bytes"),)))
    execution.engine.advance(until=(cached.event,))
    assert cached.event.time > handle.event.time


def test_fifo_repeat_arbitration_uses_public_entry_not_batch():
    engine = Engine([Resource("H", 1, unit="u", policy="fifo")])
    conductor = FragmentConductor(engine)
    a = conductor.submit(_fragment("A", (("H", 1, "u"),), repeats=2))
    b = conductor.submit(_fragment("B", (("H", 1, "u"),)))
    conductor.advance()
    assert not a.batched
    assert (a.event.time, b.event.time) == (3, 2)
    assert engine.serviced == {"H": 3.0}


def test_fair_single_resource_repeat_batches_when_equivalent():
    engine = Engine([Resource("H", 1, unit="u", policy="fair")])
    conductor = FragmentConductor(engine)
    a = conductor.submit(_fragment("A", (("H", 1, "u"),), repeats=2))
    b = conductor.submit(_fragment("B", (("H", 1, "u"),)))
    conductor.advance()
    assert a.batched
    # Batch and two sequential 1-unit fair repeats both leave B at 2 and A at 3.
    assert (a.event.time, b.event.time) == (3, 2)
    assert engine.serviced == {"H": 3.0}


def test_non_coupled_multi_resource_repeat_stays_lazy_for_late_arrival():
    engine = Engine([Resource("r1", 1, unit="u"),
                     Resource("r2", 1, unit="u")])
    conductor = FragmentConductor(engine)
    a = conductor.submit(_fragment(
        "A", (("r1", 1, "u"), ("r2", 10, "u")), repeats=2))
    conductor.advance(horizon=1.1)
    b = conductor.submit(_fragment("B", (("r1", 1, "u"),)))
    conductor.advance()
    assert not a.batched
    # A's first repeat has already finished r1 at 1, so B owns r1 from 1.1.
    # Batching A's two repeats would keep r1 occupied and finish B near 3.0.
    assert b.event.time == pytest.approx(2.1)
    assert a.event.time == pytest.approx(20.0)
    assert engine.serviced == {"r1": 3.0, "r2": 20.0}


def test_coupled_multi_resource_repeat_is_equivalent_when_all_fair():
    engine = Engine([Resource("r1", 10, unit="u"),
                     Resource("r2", 10, unit="u")])
    conductor = FragmentConductor(engine)
    handle = conductor.submit(_fragment(
        "A", (("r1", 10, "u"), ("r2", 20, "u")),
        repeats=2, coupled=True))
    conductor.advance()
    assert handle.batched
    assert handle.event.time == pytest.approx(4.0)
    assert engine.serviced == {"r1": 20.0, "r2": 40.0}


def test_lazy_fifo_repeats_keep_only_active_work():
    engine = Engine([Resource("H", 1, unit="u", policy="fifo")])
    conductor = FragmentConductor(engine)
    handle = conductor.submit(_fragment("A", (("H", 1, "u"),), repeats=2000))
    conductor.advance()
    assert not handle.batched
    assert handle.event.time == pytest.approx(2000.0)
    assert engine.serviced == {"H": 2000.0}
    assert engine.summary()["peak_live_tasks"] <= 2
    assert engine.summary()["live_tasks"] == 0


def test_independent_program_holds_release_at_their_own_events():
    engine = Engine([Resource("H", 1, unit="u")], pools={"P": 128})
    conductor = FragmentConductor(engine)
    hold_a = engine.submit("hold_a", latency=3)
    hold_b = engine.submit("hold_b", latency=7)
    handle = conductor.submit(
        _fragment("two_holds", (("H", 1, "u"),),
                  allocations=(AllocationEffect.acquire(
                                   "a", "P", 32, lifetime="program"),
                               AllocationEffect.acquire(
                                   "b", "P", 32, lifetime="program"))),
        holds={"a": hold_a, "b": hold_b})
    releases = dict(handle.release_by_name)
    conductor.advance(until=(releases["a"], releases["b"]))
    assert releases["a"].time == 3
    assert releases["b"].time == 7
    assert engine.allocated == {"P": 0.0}


def test_local_program_release_is_not_delayed_by_unrelated_hold():
    engine = Engine([Resource("H", 1, unit="u")], pools={"P": 128})
    conductor = FragmentConductor(engine)
    hold_b = engine.submit("hold_b", latency=7)
    handle = conductor.submit(
        _fragment("mixed", (("H", 1, "u"),),
                  allocations=(
                      AllocationEffect.acquire("a", "P", 32,
                                               lifetime="program"),
                      AllocationEffect.release("a", "P", 32,
                                               lifetime="program"),
                      AllocationEffect.acquire("b", "P", 32,
                                               lifetime="program"))),
        holds={"b": hold_b})
    releases = dict(handle.release_by_name)
    conductor.advance(until=(releases["a"], releases["b"]))
    assert releases["a"].time == 1
    assert releases["b"].time == 7
    assert engine.allocated == {"P": 0.0}


def test_event_port_kinds_bind_grant_release_and_complete():
    engine = Engine([Resource("H", 1, unit="u")], pools={"P": 128})
    conductor = FragmentConductor(engine)
    handle = conductor.submit(_fragment(
        "event_kinds", (("H", 1, "u"),),
        allocations=(AllocationEffect.acquire("staging", "P", 32,
                                              lifetime="program"),
                     AllocationEffect.release("staging", "P", 32,
                                              lifetime="program")),
        ports=PortSet(event_out=(
            EventPortSpec("staging_grant", "out", "grant"),
            EventPortSpec("staging_release", "out", "release"),
            EventPortSpec("done", "out", "complete")))))
    conductor.advance()
    assert conductor.event("staging_grant").time == 0
    assert conductor.event("staging_release").time == 1
    assert conductor.event("done") is handle.event
    assert handle.grant_by_name[0][1].time == 0
    assert handle.release_by_name[0][1].time == 1


def test_ambiguous_event_port_is_rejected_before_submission():
    engine = Engine([Resource("H", 1, unit="u")], pools={"P": 128})
    conductor = FragmentConductor(engine)
    fragment = _fragment(
        "ambiguous_event", (("H", 1, "u"),),
        allocations=(AllocationEffect.acquire("a", "P", 32,
                                              lifetime="program"),
                     AllocationEffect.release("a", "P", 32,
                                              lifetime="program"),
                     AllocationEffect.acquire("b", "P", 32,
                                              lifetime="program"),
                     AllocationEffect.release("b", "P", 32,
                                              lifetime="program")),
        ports=PortSet(event_out=(
            EventPortSpec("unmatched", "out", "release"),)))
    with pytest.raises(UnsupportedFragmentSemantics,
                       match="fragment_event_port_ambiguous:release:unmatched"):
        conductor.submit(fragment)
    assert engine.summary()["live_tasks"] == 0
    assert engine.allocated == {"P": 0.0}


def test_record_event_port_requires_explicit_binding_or_rejects():
    engine = Engine([Resource("H", 1, unit="u")])
    conductor = FragmentConductor(engine)
    fragment = _fragment(
        "record_event", (("H", 1, "u"),),
        ports=PortSet(event_out=(EventPortSpec("recorded", "out", "record"),)))
    with pytest.raises(
            UnsupportedFragmentSemantics,
            match="fragment_event_port_requires_explicit_binding:record:recorded"):
        conductor.submit(fragment)
    marked = engine.submit("marked", latency=3)
    handle = conductor.submit(fragment, event_bindings={"recorded": marked})
    conductor.advance(until=(handle.event,))
    assert conductor.event("recorded") is marked


def test_policy_binding_mismatch_is_rejected():
    fifo = Engine([Resource("H", 1, unit="u", policy="fifo")])
    with pytest.raises(FragmentContractError,
                       match="fragment_resource_policy_mismatch:H:H:fair:fifo"):
        FragmentConductor(fifo).submit(_fragment(
            "fair_on_fifo", (("H", 1, "u"),),
            policies=(("H", ResourcePolicy(sharing="fair")),)))

    fair = Engine([Resource("H", 1, unit="u", policy="fair")])
    with pytest.raises(FragmentContractError,
                       match="fragment_resource_policy_mismatch:H:H:fifo:fair"):
        FragmentConductor(fair).submit(_fragment(
            "fifo_on_fair", (("H", 1, "u"),),
            policies=(("H", ResourcePolicy(sharing="fifo")),)))


def test_work_unit_mismatch_and_explicit_conversion():
    engine = Engine([Resource("H", 8, unit="bytes")])
    conductor = FragmentConductor(engine)
    with pytest.raises(FragmentContractError,
                       match="fragment_work_unit_mismatch:H:cycles:bytes"):
        conductor.submit(_fragment("cycles", (("H", 8, "cycles"),)))

    converted = FragmentConductor(
        engine, unit_conversions={("cycles", "bytes"): 8})
    handle = converted.submit(_fragment("converted", (("H", 16, "cycles"),)))
    converted.advance()
    assert handle.event.time == pytest.approx(0.25)
    assert engine.serviced == {"H": 2.0}


def test_coupled_fifo_priority_is_consistent_and_non_preemptive():
    def coupled(name, priority=0):
        return _fragment(name, (("q", 1, "u"), ("r", 10, "u")),
                         coupled=True, priority=priority)

    engine = Engine([Resource("q", 1, unit="u", policy="fifo"),
                     Resource("r", 10, unit="u")])
    conductor = FragmentConductor(engine)
    a = conductor.submit(coupled("A", priority=1))
    b = conductor.submit(coupled("B", priority=0))
    conductor.advance()
    assert (a.event.time, b.event.time) == (2, 1)
    assert engine.serviced == {"q": 2.0, "r": 20.0}

    engine = Engine([Resource("q", 1, unit="u", policy="fifo"),
                     Resource("r", 10, unit="u")])
    conductor = FragmentConductor(engine)
    a = conductor.submit(coupled("A"))
    conductor.advance(horizon=0.5)
    b = conductor.submit(coupled("B", priority=0))
    conductor.advance()
    # A already owns FIFO q; the later higher-priority task waits.
    assert (a.event.time, b.event.time) == (1, 2)


def test_program_allocations_share_one_admission_boundary_without_deadlock():
    def pq():
        return (AllocationEffect.acquire("p", "P", 1, lifetime="program"),
                AllocationEffect.acquire("q", "Q", 1, lifetime="program"),
                AllocationEffect.release("p", "P", 1, lifetime="program"),
                AllocationEffect.release("q", "Q", 1, lifetime="program"))

    engine = Engine([Resource("run", 1, unit="u")],
                    pools={"P": 1, "Q": 1})
    conductor = FragmentConductor(engine)
    free_q = engine.submit("free_q", latency=10)
    blocker = conductor.submit(
        _fragment("blocker", allocations=(AllocationEffect.acquire(
            "q", "Q", 1, lifetime="program"),)),
        holds={"q": free_q})
    a = conductor.submit(_fragment(
        "A", (("run", 1, "u"),), allocations=pq(), priority=1))
    b = conductor.submit(_fragment(
        "B", (("run", 1, "u"),), allocations=pq(), priority=0))
    conductor.advance()
    assert blocker.release_events[0].time == 10
    # B (priority 0) takes P+Q together at 10; A cannot half-admit and hold P.
    assert (b.event.started, b.event.time) == (10, 11)
    assert (a.event.started, a.event.time) == (11, 12)
    assert dict(b.grant_by_name)["p"].time == dict(b.grant_by_name)["q"].time == 10
    assert dict(a.grant_by_name)["p"].time == dict(a.grant_by_name)["q"].time == 11
    assert engine.serviced == {"run": 2.0}
    assert engine.allocated == {"P": 0.0, "Q": 0.0}
    assert engine.summary()["live_tasks"] == 0


def test_execution_exposes_declared_and_fallback_request_commits():
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import IterationWork, OpSpec
    from mindie_sim.schema.workload import IterationPlan, RequestView
    from test_prediction_system import case

    parsed = parse_case(case())
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    plan = IterationPlan((
        RequestView("r0", 1, 2, True, "decode"),
        RequestView("r1", 1, 2, True, "decode"),
    ))
    work = OpSpec(
        "work", "test", "decode", 0, 0, ("r0", "r1"), (1,), (1,), (1,), (1,),
        hbm_bytes=1000, dtype="bf16")
    commit = OpSpec(
        "commit", "token_commit", "decode", 0, -1, ("r0",), (1,), (1,), (1,),
        (1,), deps=("work",), recipe="token_commit")
    step = execution.add(
        IterationWork((work, commit), plan, 1, 2, 2, None, "decode", "test"),
        0, 0)
    execution.engine.advance(until=(step.completion,), first=False)
    assert set(step.declared_commits) == {"r0"}
    assert set(step.fallback_commits) == {"r1"}
    assert set(step.commits) == {"r0", "r1"}
    assert step.declared_commits["r0"] is not step.fallback_commits["r1"]
    payload = step.to_dict()
    assert set(payload["declared_commit_ms"]) == {"r0"}
    assert set(payload["fallback_commit_ms"]) == {"r1"}


def test_event_batch_preflight_is_atomic_before_engine_or_registry():
    engine = Engine([Resource("H", 1, unit="u")])
    conductor = FragmentConductor(engine, event_limit=2)
    old = engine.submit("old")
    engine.advance(until=(old,))
    conductor.record_event("old", old)
    fragment = _fragment(
        "too_many_ports", (("H", 1, "u"),),
        ports=PortSet(event_out=(
            EventPortSpec("x", "out", "complete"),
            EventPortSpec("y", "out", "complete"))))
    with pytest.raises(FragmentContractError,
                       match="fragment_event_name_limit_exceeded:2"):
        conductor.submit(fragment)
    assert conductor.event("old") is old
    for name in ("x", "y"):
        with pytest.raises(FragmentContractError, match=f"fragment_event_unbound:{name}"):
            conductor.event(name)
    assert conductor.summary()["live_fragments"] == 0
    assert conductor.summary()["totals"]["fragments"] == 0
    assert engine.summary()["live_tasks"] == 0
    assert engine.serviced == {"H": 0.0}
    engine.advance()
    assert engine.serviced == {"H": 0.0}

    # Exactly fitting union (old + one new name) still publishes normally.
    handle = conductor.submit(_fragment(
        "fits_one", (("H", 1, "u"),),
        ports=PortSet(event_out=(
            EventPortSpec("x", "out", "complete"),))))
    conductor.advance()
    assert handle.event.time == 1
    assert conductor.event("old") is old
    assert conductor.event("x").time == 1
    assert engine.serviced == {"H": 1.0}


def test_event_batch_preflight_checks_generation_and_binding_atomically():
    engine = Engine([Resource("H", 1, unit="u")])
    conductor = FragmentConductor(engine, event_limit=8)
    old = engine.submit("old")
    engine.advance(until=(old,))
    conductor.record_event("x", old, generation=0)
    duplicate = _fragment(
        "duplicate_generation", (("H", 1, "u"),),
        ports=PortSet(event_out=(
            EventPortSpec("x", "out", "complete", generation=0),)))
    with pytest.raises(
            FragmentContractError,
            match="fragment_event_generation_already_recorded:x:0"):
        conductor.submit(duplicate)
    assert conductor.event("x", 0) is old
    assert engine.serviced == {"H": 0.0}
    assert engine.summary()["live_tasks"] == 0

    bad_binding = _fragment(
        "bad_binding", (("H", 1, "u"),),
        ports=PortSet(event_out=(
            EventPortSpec("y", "out", "record"),)))
    with pytest.raises(FragmentContractError,
                       match="fragment_event_binding_invalid:event:y"):
        conductor.submit(bad_binding, event_bindings={"y": object()})
    with pytest.raises(FragmentContractError, match="fragment_event_unbound:y"):
        conductor.event("y")
    assert engine.serviced == {"H": 0.0}


def test_single_event_register_does_not_evict_before_name_rejection():
    engine = Engine(())
    conductor = FragmentConductor(engine, event_limit=2)
    old = engine.submit("old")
    other = engine.submit("other")
    engine.advance(until=(old, other))
    conductor.record_event("old", old)
    conductor.record_event("other", other)
    assert conductor.summary()["recorded_events"] == 2
    with pytest.raises(FragmentContractError,
                       match="fragment_event_name_limit_exceeded:2"):
        conductor.record_event("z", old)
    assert conductor.event("old") is old
    assert conductor.event("other") is other
    with pytest.raises(FragmentContractError, match="fragment_event_unbound:z"):
        conductor.event("z")
    assert conductor.summary()["recorded_events"] == 2
    assert conductor.summary()["events_evicted"] == 0


def test_graph_replay_registers_one_host_submit_handle_per_real_launch():
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import IterationWork, OpSpec
    from mindie_sim.schema.workload import IterationPlan, RequestView
    from test_prediction_system import case

    raw = case()
    raw["runtime"].update(
        graph_mode="auto", graph_capture_keys=[4], graph_capture_mode="full",
        submission={"graph_replay_ms": .25})
    parsed = parse_case(raw)
    execution = Execution(parsed, CostModel(parsed))
    plan = IterationPlan((RequestView("r", 1, 4, True, "decode"),))
    ops = tuple(OpSpec(
        f"op{index}", "test", "decode", 0, 0, ("r",), (1,), (1,), (1,), (1,),
        hbm_bytes=1000, dtype="bf16") for index in range(5))
    step = execution.add(
        IterationWork(ops, plan, 1, 1, 1, 4, "decode", "test"), 0, 0)
    assert execution.submission_counts == {
        "eager_invocations": 0, "graph_replays": 1}
    assert len(step.host_submits) == 1
    execution.engine.advance(until=(step.completion,), first=False)
    assert step.host_submits[0][2].time == 0
    assert len(step.to_dict()["host_submits"]) == 1


def test_eager_launches_at_the_same_time_are_kept_distinct():
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import IterationWork, OpSpec
    from mindie_sim.schema.workload import IterationPlan, RequestView
    from test_prediction_system import case

    raw = case(pp=2)
    parsed = parse_case(raw)
    execution = Execution(parsed, CostModel(parsed))
    plan = IterationPlan((RequestView("r", 1, 4, True, "decode"),))
    ops = tuple(OpSpec(
        f"op{stage}", "test", "decode", stage, 0, ("r",), (1,), (1,), (1,),
        (1,), hbm_bytes=1000, dtype="bf16") for stage in range(2))
    step = execution.add(
        IterationWork(ops, plan, 1, 1, 1, None, "decode", "test"), 0, 0)
    assert execution.submission_counts == {
        "eager_invocations": 2, "graph_replays": 0}
    assert len(step.host_submits) == 2
    handles = [event for _, _, event in step.host_submits]
    assert len({id(event) for event in handles}) == 2
    assert len({event.sequence for event in handles}) == 2
    execution.engine.advance(until=(step.completion,), first=False)
    assert [event.time for event in handles] == [0, 0]


def _pipeline_tile(index):
    from mindie_sim.prediction.program import Stage
    return (
        Stage("load", {"r": 1.0},
              acquire=(Allocation(f"panel{index}", "p", 1, (0,)),)),
        Stage("compute", {"r": 1.0}, after=("load",),
              free=(Allocation(f"panel{index}", "p", 1, (0,)),)),
    )


def test_pipeline_template_consumer_matches_detailed_competition_controls():
    from mindie_sim.prediction.program import PipelineProgram, Stage
    from mindie_sim.prediction.scheduling import start_pipeline_template

    engine = Engine([Resource("r", 1.0)], pools={"p": 2}, pool_units={"p": 1})
    template = PipelineProgram(8, 2, _pipeline_tile, homogeneous=True).template()
    done = start_pipeline_template(engine, template, "p", engine.submit("ready"))
    external = engine.submit("external", {"r": 4.0}, release=0)
    successor = engine.submit("successor", after=(external,), release=0)
    engine.advance(until=(done, external, successor), first=False)
    assert engine.serviced == {"r": 20.0}
    assert (done.time, external.time, successor.time) == (20, 11, 11)

    engine = Engine([Resource("r", 1.0)], pools={"p": 2}, pool_units={"p": 1})
    held = Allocation("external", "p", 1, (0,))
    hold = engine.submit("external-hold", acquire=(held,), release=0)
    template = PipelineProgram(8, 2, _pipeline_tile, homogeneous=True).template()
    done = start_pipeline_template(engine, template, "p", engine.submit("ready"))
    release = engine.submit("external-release", after=(done,), free=(held,))
    engine.advance(until=(hold, done, release), first=False)
    assert done.done and release.done
    assert not engine.allocated["p"]

    def tail(index):
        size = 1 if index == 0 else 3
        name = f"tail{index}"
        return (
            Stage("load", {"r": 1.0},
                  acquire=(Allocation(name, "p", size, (0,)),)),
            Stage("compute", {"r": 0.0}, after=("load",),
                  free=(Allocation(name, "p", size, (0,)),)),
        )

    engine = Engine([Resource("r", 1.0)], pools={"p": 10}, pool_units={"p": 1})
    template = PipelineProgram(2, 2, tail, homogeneous=True).template()
    done = start_pipeline_template(engine, template, "tail", engine.submit("ready"))
    engine.advance(until=(done,))
    assert engine.peak_allocated["p"] == 4


def test_pipeline_template_phase_counts_preserve_repeat_boundaries():
    from mindie_sim.prediction.program import PipelineProgram
    from mindie_sim.prediction.scheduling import start_pipeline_template

    engine = Engine([Resource("r", 1.0)], pools={"p": 2}, pool_units={"p": 1})
    template = PipelineProgram(8, 2, _pipeline_tile, homogeneous=True).template(panels_per_output=4)
    assert [phase.kind for phase in template.phases] == [
        "first", "middle", "last", "first", "middle", "last"]
    assert [phase.count for phase in template.phases] == [1, 2, 1, 1, 2, 1]
    done = start_pipeline_template(engine, template, "p", engine.submit("ready"))
    external = engine.submit("external", {"r": 4.0}, release=0)
    successor = engine.submit("successor", after=(external,), release=0)
    engine.advance(until=(done, external, successor), first=False)
    assert (done.time, external.time, successor.time) == (20, 11, 11)
    assert engine.serviced == {"r": 20.0}


def test_pipeline_template_consumer_is_lazy_for_many_tiles():
    from mindie_sim.prediction.program import PipelineProgram, Stage
    from mindie_sim.prediction.scheduling import start_pipeline_template

    def tile(index):
        return (Stage("work", {"r": 1.0}),)

    engine = Engine([Resource("r", 1.0)])
    template = PipelineProgram(65, 2, tile, homogeneous=True).template()
    template = type(template)(template.owner, template.slots, template.phases,
                              setup_ms=1.0)
    done = start_pipeline_template(engine, template, "p", engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == 66
    assert engine.serviced == {"r": 65.0}
    assert engine.summary()["peak_live_tasks"] < 10
    assert engine.summary()["live_tasks"] == 0


def test_execution_ordinary_production_does_not_inject_primitive_floor():
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.primitives import PrimitiveTiming
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import OpSpec
    from test_prediction_system import case

    parsed = parse_case(case())
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    op = OpSpec("x", "test", "decode", 0, 0, ("r",), (1,), (1,), (1,), (1,),
                hbm_bytes=1000, dtype="bf16")

    diagnostic = {"timing": PrimitiveTiming(5.0, 1, 5.0, 5.0, ())}
    execution.cost.lookup_diagnostic = lambda operator, scope: diagnostic
    production = execution._context(op, 0, "none")
    assert production.minimum_ms == 0
    assert production.work["memory@r0:s0"] == 1000.0

    execution.legacy_diagnostics = True
    legacy = execution._context(op, 0, "none")
    assert legacy.minimum_ms == 5.0


def test_execution_stage_conductors_share_one_visible_state_ledger():
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    from test_prediction_system import case

    parsed = parse_case(case(pp=2))
    execution = Execution(parsed, CostModel(parsed), include_submission=False,
                          fragment_state={"kv": 7})
    first = execution.fragment_conductor(replica=0, stage=0)
    second = execution.fragment_conductor(replica=0, stage=1)
    assert first.state is second.state is execution.fragment_state
    write = first.submit(_fragment(
        "write", (("hbm", 1_000_000, "bytes"),),
        state_effects=(StateEffect("kv", "write", version=8),)))
    commit = first.submit(_fragment(
        "commit", state_effects=(StateEffect("kv", "commit", version=8),)),
        after=(write.event,))
    read = second.submit(_fragment(
        "read", (("hbm", 1_000_000, "bytes"),),
        state_effects=(StateEffect("kv", "read", version=8),)),
        after=(commit.event,))
    execution.engine.advance(until=(read.event,), first=False)
    assert first.state.committed_version("kv") == 8
    assert second.state is first.state
    assert read.event.done


def test_execution_ordinary_production_does_not_inject_response_duration_floor():
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import OpSpec
    from test_prediction_system import case

    class Response:
        mode = "declared_service"
        duration_ms = 5.0

        def diagnostics(self):
            return {"mode": self.mode, "response_us": 5000.0}

    parsed = parse_case(case())
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    op = OpSpec("x", "test", "decode", 0, 0, ("r",), (1,), (1,), (1,), (1,),
                hbm_bytes=1000, dtype="bf16")
    calls = []

    def resolve(context, recorder):
        calls.append(1)
        return Response()

    execution.regimes.resolve = resolve
    production = execution._context(op, 0, "none")
    assert production.minimum_ms == 0
    assert production.service_regime is None
    assert calls == []

    execution.legacy_diagnostics = True
    legacy = execution._context(op, 0, "none")
    assert legacy.minimum_ms == 5.0
    assert legacy.service_regime["mode"] == "declared_service"
    assert calls == [1]


def test_pipeline_template_preserves_cross_tile_resident_allocations():
    from mindie_sim.prediction.program import (
        PipelinePhase, PipelineTemplate, Stage,
    )
    from mindie_sim.prediction.scheduling import start_pipeline_template

    def first():
        return (
            Stage("load", {"r": 1.0}, acquire=(
                Allocation("panel0", "p", 1, (0,)),
                Allocation("accumulator0", "p", 1, (0,)))),
            Stage("compute", {"r": 1.0}, after=("load",), free=(
                Allocation("panel0", "p", 1, (0,)),)),
        )

    def middle(index):
        return (
            Stage("load", {"r": 1.0}, acquire=(
                Allocation(f"panel{index}", "p", 1, (0,)),)),
            Stage("compute", {"r": 1.0}, after=("load",), free=(
                Allocation(f"panel{index}", "p", 1, (0,)),)),
        )

    def last():
        return (
            Stage("load", {"r": 1.0}, acquire=(
                Allocation("panel4", "p", 1, (0,)),)),
            Stage("compute", {"r": 1.0}, after=("load",), free=(
                Allocation("panel4", "p", 1, (0,)),
                Allocation("accumulator0", "p", 1, (0,)))),
        )

    template = PipelineTemplate(0, 2, (
        PipelinePhase("first", first(), 1),
        PipelinePhase("middle", middle(1), 3),
        PipelinePhase("last", last(), 1),
    ))
    engine = Engine([Resource("r", 1.0)], pools={"p": 4}, pool_units={"p": 1})
    done = start_pipeline_template(engine, template, "m", engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == 10
    assert engine.allocated == {"p": 0.0}
    # accumulator0 stays live from tile0 to tile4; peak is not slots*max.
    assert engine.peak_allocated["p"] == 3


def test_execution_applies_explicit_t06_resource_service_once():
    from types import SimpleNamespace
    from mindie_sim.observations.resource_services import normalize_resource_service
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.program import KernelProgram, Stage
    from mindie_sim.prediction.structure import parse_case
    from test_prediction_system import case

    record = {
        "id": "svc", "scope": "independent_resource", "resource": "hbm",
        "phase": "decode", "work_unit": "bytes",
        "boundary": "resource_phase", "startup_us": 1000, "response_us": 0,
        "rate_scope": "aggregate", "active_units": 1,
        "supply": {"kind": "absolute_rate", "rate": 1_000_000.0},
        "conditions": {}, "identification": {},
    }
    service = normalize_resource_service(record)
    op = SimpleNamespace(metadata=(("resource_service", {
        "service_id": "svc", "resource_role": "hbm",
        "work_unit": "bytes", "phase": "decode"}),))
    context = SimpleNamespace(
        op=op, resource_names={"hbm": "memory"},
        work_units={"hbm": "bytes"})
    parsed = parse_case(case())
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    execution.resource_services = {"svc": service}
    wrapped = execution._apply_resource_service_program(
        context, KernelProgram((Stage("work", {"memory": 1000}),)))
    engine = Engine([Resource("memory", 1000, unit="bytes")])
    done = wrapped.start(engine, "svc", engine.submit("ready"))
    engine.advance(until=(done,))
    # fixed 1ms setup + 1000B at 1000B/ms; no whole-call floor.
    assert done.time == 2
    assert engine.serviced == {"memory": 1000.0}


def test_pipeline_template_quiescent_homogeneous_batch_preserves_work():
    from mindie_sim.prediction.program import PipelineProgram, Stage
    from mindie_sim.prediction.scheduling import start_pipeline_template

    def tile(index):
        return (Stage("work", {"r": 1.0}),)

    engine = Engine([Resource("r", 1.0)])
    template = PipelineProgram(65, 2, tile, homogeneous=True).template()
    template = type(template)(template.owner, template.slots, template.phases,
                              setup_ms=1.0)
    done = start_pipeline_template(
        engine, template, "p", engine.submit("ready"), quiescent=True)
    engine.advance(until=(done,))
    assert done.time == 66
    assert engine.serviced == {"r": 65.0}
    assert engine.completed_count <= 10
    assert engine.summary()["live_tasks"] == 0


def test_pipeline_template_setup_body_drain_share_one_allocation_binding():
    from mindie_sim.prediction.program import PipelinePhase, PipelineTemplate, Stage
    from mindie_sim.prediction.scheduling import start_pipeline_template

    setup = (Stage("setup", acquire=(Allocation("w", "p", 1, (0,)),)),)
    body = (Stage("work", {"r": 1.0}),)
    drain = (Stage("drain", free=(Allocation("w", "p", 1, (0,)),)),)
    template = PipelineTemplate(0, 1, (PipelinePhase("tile", body, 1),),
                                setup=setup, drain=drain)
    engine = Engine([Resource("r", 1.0)], pools={"p": 1}, pool_units={"p": 1})
    done = start_pipeline_template(engine, template, "p", engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == 1
    assert engine.serviced == {"r": 1.0}
    assert engine.allocated == {"p": 0.0}


def test_pipeline_template_empty_body_still_runs_drain_and_release():
    from mindie_sim.prediction.program import PipelineTemplate, Stage
    from mindie_sim.prediction.scheduling import start_pipeline_template

    setup = (Stage("setup", acquire=(Allocation("w", "p", 1, (0,)),)),)
    drain = (Stage("drain", {"r": 1.0}, free=(Allocation("w", "p", 1, (0,)),)),)
    template = PipelineTemplate(0, 1, (), setup=setup, drain=drain)
    engine = Engine([Resource("r", 1.0)], pools={"p": 1}, pool_units={"p": 1})
    done = start_pipeline_template(engine, template, "p", engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == 1
    assert engine.serviced == {"r": 1.0}
    assert engine.allocated == {"p": 0.0}


def test_pipeline_template_invocation_root_keeps_concurrent_instances_distinct():
    from mindie_sim.prediction.program import PipelinePhase, PipelineTemplate, Stage
    from mindie_sim.prediction.scheduling import start_pipeline_template

    def instance(name, allocation):
        stage = Stage("work", {"r": 2.0},
                      acquire=(Allocation(allocation, "p", 1, (0,)),))
        drain = Stage("drain", free=(Allocation(allocation, "p", 1, (0,)),))
        return PipelineTemplate(
            0, 1, (PipelinePhase("tile", (stage,), 1),), drain=(drain,))

    engine = Engine([Resource("r", 1.0)], pools={"p": 2}, pool_units={"p": 1})
    one = start_pipeline_template(
        engine, instance("tile0", "w0"), "outer:tile0", engine.submit("ready"))
    two = start_pipeline_template(
        engine, instance("tile1", "w1"), "outer:tile1", engine.submit("ready"))
    engine.advance(until=(one, two), first=False)
    assert one.time == two.time == 4
    assert engine.serviced["r"] == 4.0
    assert engine.peak_allocated["p"] == 2
    assert engine.allocated == {"p": 0.0}


def test_execution_binds_one_transport_options_to_context_and_occurrence():
    from types import SimpleNamespace
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    from mindie_sim.schema.operators import OpSpec
    from test_prediction_system import case

    parsed = parse_case(case())
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    op = OpSpec("x", "alltoall", "decode", 0, 0, ("r",), (1,), (1,),
                (1,), (1,), link_bytes=128, dtype="bf16")
    recipe = SimpleNamespace(
        algorithm="direct_peer",
        plan_options={"max_inflight_per_rank": 2, "chunk_bytes": 64})
    context = execution._context(op, 0, "none", recipe=recipe)
    stored = execution.transport_options_for(op, 0, "none")
    assert stored.max_inflight_per_rank == 2
    assert stored.chunk_bytes == 64
    assert context.transport_options is stored


def test_engine_periodic_apply_63_cycle_positive_control():
    from mindie_sim.prediction.engine_periodic import (
        CycleCorrespondence, CycleSkip, IntervalDelta, RunCorrespondence,
    )

    engine = Engine([Resource("mte2", 1.0), Resource("cube", 1.0)],
                    pools={"l1": 100}, pool_units={"l1": 1})
    engine._periodic_runs[2] = {
        "run_id": 2, "phase_ordinal": 5, "phase_end": 67,
        "next": 4, "finished": 2, "slots": (0, 1),
        "live_tiles": {2, 3}, "retire_tiles": {900: 2},
    }
    interval = IntervalDelta(
        dt=1.0, issued_retired_by_run={2: (1, 1)}, sequence_count=4,
        service_by_resource={"mte2": 8.0, "cube": 16.0},
        busy_by_resource={"mte2": 1.0, "cube": 1.0},
        allocation_event_count=2, per_unit_peaks={"l1": 8.0},
        logical_completions=1, crossed_external_or_opaque=False)
    run = RunCorrespondence(
        2, 5, 67, 4, 67, 2, 65, tile_offset=63, finished_offset=63)
    correspondence = CycleCorrespondence(
        (run,), (), 252, {2: 63})
    skip = CycleSkip(63, 1.0, {2: 1}, 4, interval, correspondence)
    engine._apply_periodic_skip(skip)
    assert engine.now == 63
    assert engine._sequence == 252
    assert engine.serviced == {"mte2": 504.0, "cube": 1008.0}
    assert engine.busy_ms == {"mte2": 63.0, "cube": 63.0}
    assert engine.allocation_event_count == 126
    assert engine.peak_allocated["l1"] == 8.0
    assert engine._periodic_runs[2]["next"] == 67
    assert engine._periodic_runs[2]["finished"] == 65
    assert engine._periodic_runs[2]["live_tiles"] == {65, 66}
    assert engine._periodic_runs[2]["slots"] == (0, 1)


def test_periodic_skip_resumes_real_scheduler_retire_and_tail_once():
    from mindie_sim.prediction.engine_periodic import (
        CycleCorrespondence, CycleSkip, IntervalDelta, RunCorrespondence,
    )
    from mindie_sim.prediction.program import (
        PipelinePhase, PipelineTemplate, Stage,
    )
    from mindie_sim.prediction.scheduling import start_pipeline_template

    body = Stage("work", {"r": 1.0})
    tail = Stage("tail", {"r": 1.0})
    template = PipelineTemplate(0, 2, (
        PipelinePhase("body", (body,), 67),
        PipelinePhase("tail", (tail,), 1)))
    engine = Engine([Resource("r", 1.0)])
    done = start_pipeline_template(engine, template, "m", engine.submit("ready"))
    stop = engine.submit("stop", latency=2)
    engine.advance(until=(stop,))
    run_id = next(iter(engine._periodic_runs))
    run = engine._periodic_runs[run_id]
    assert (run["next"], run["finished"], run["phase_end"]) == (4, 2, 67)
    assert run["live_tiles"] == {2, 3}
    interval = IntervalDelta(
        1.0, {run_id: (1, 1)}, 4, {"r": 1.0}, {"r": 1.0}, 2,
        {"p": 7.0}, 1, False)
    correspondence = CycleCorrespondence(
        (RunCorrespondence(run_id, 0, 67, 4, 67, 2, 65, 63, 63),), (),
        252, {run_id: 63})
    engine._apply_periodic_skip(CycleSkip(
        63, 1.0, {run_id: 1}, 4, interval, correspondence))
    assert run["live_tiles"] == {65, 66}
    engine.advance(until=(done,), max_steps=100)
    assert done.time == 68
    assert engine.serviced == {"r": 68.0}
    assert engine.summary()["live_tasks"] == 0
    assert not engine._periodic_runs
    assert not engine._periodic_tasks


def test_execution_and_conductor_share_t08_memory_grant_until_release():
    from dataclasses import replace
    from mindie_sim.prediction.capacity import Memory, MemoryPolicy
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    from test_prediction_system import case

    raw = case()
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory={"derive_workspace": False})
    parsed = parse_case(raw)
    base = Memory(parsed)
    capacity = int(base.weights[0] + 12288)
    tight = replace(parsed, machine=replace(
        parsed.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in parsed.machine.dies)))
    memory = Memory(tight, MemoryPolicy(derive_workspace=False))
    engine = Engine([Resource("r", 1.0),
                     Resource("hbm", 1e9, unit="bytes")],
                    pools={"hbm": 1e9}, pool_units={"hbm": 1})
    execution = Execution(parsed, CostModel(parsed), engine=engine,
                          include_submission=False, memory=memory)
    conductor = execution.fragment_conductor(replica=0, stage=0)
    fragments = [
        _fragment(f"f{index}", (("r", 1.0, "work"),),
                  allocations=(
                      AllocationEffect.acquire("a", "hbm", 4096),
                      AllocationEffect.release("a", "hbm", 4096)))
        for index in range(4)]
    handles = [conductor.submit(item, owner="x") for item in fragments]
    engine.advance(until=tuple(handle.event for handle in handles),
                   first=False, max_steps=100)
    times = [handle.event.time for handle in handles]
    assert times[:3] == [pytest.approx(3)] * 3
    assert times[3] == pytest.approx(4)
    assert memory.live_grants(0, 0) == 0.0
    assert memory.peak_combined[0][0] == 12288.0


def test_quiescent_batch_uses_same_terminal_drain_and_cleanup():
    from mindie_sim.prediction.program import (
        PipelinePhase, PipelineTemplate, Stage,
    )
    from mindie_sim.prediction.scheduling import start_pipeline_template

    setup = (Stage("setup", acquire=(Allocation("w", "p", 1, (0,)),)),)
    body = (Stage("work", {"r": 1.0}),)
    drain = (Stage("drain", {"r": 1.0},
                   free=(Allocation("w", "p", 1, (0,)),)),)
    template = PipelineTemplate(
        0, 1, (PipelinePhase("body", body, 2),),
        setup=setup, drain=drain)
    engine = Engine([Resource("r", 1.0)], pools={"p": 1}, pool_units={"p": 1})
    done = start_pipeline_template(
        engine, template, "p", engine.submit("ready"), quiescent=True)
    engine.advance(until=(done,))
    assert done.time == 3
    assert engine.serviced == {"r": 3.0}
    assert engine.allocated == {"p": 0.0}
    assert not engine._periodic_runs
    assert not engine._periodic_tasks


def test_conductor_async_reader_bridge_uses_same_memory_generation():
    from mindie_sim.prediction.capacity import Memory, MemoryPolicy
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.structure import parse_case
    from test_prediction_system import case

    parsed = parse_case(case())
    memory = Memory(parsed, MemoryPolicy(derive_workspace=False))
    assert memory.resize(0, "request", 1)
    execution = Execution(parsed, CostModel(parsed),
                          include_submission=False, memory=memory)
    conductor = execution.fragment_conductor(replica=0, stage=0)
    assert conductor.hold_async_reader(
        "request", "reader:0", storage="kv/page/0", ranges=(0, 32),
        generation=3)
    assert memory.pending_async_consumers(0, "request") == ("reader:0",)
    assert conductor.release_async_reader("request", "reader:0")
    assert memory.pending_async_consumers(0, "request") == ()


def test_pipeline_template_reuses_allocation_free_prepared_stage():
    from unittest.mock import patch
    from mindie_sim.prediction.program import PipelineProgram, Stage
    from mindie_sim.prediction.scheduling import start_pipeline_template

    calls = []
    original = Stage.prepare

    def counting(self, engine, name, scoped):
        calls.append(self.name)
        return original(self, engine, name, scoped)

    def tile(index):
        return (Stage("work", {"r": 1.0}),)

    template = PipelineProgram(65, 2, tile, homogeneous=True).template()
    engine = Engine([Resource("r", 1.0)])
    with patch.object(Stage, "prepare", counting):
        done = start_pipeline_template(engine, template, "p",
                                       engine.submit("ready"))
        engine.advance(until=(done,))
    assert done.time == 65
    assert calls == ["work"]


# ---------------------------------------------------------------------------
# r18: normal Execution.add state/Memory binding, real grant lifetime and
# per-reader offload. These use the public add/program entry, not a manually
# driven hold helper.

def _normal_case_with_memory(extra, *, ratios=None):
    from dataclasses import replace
    from mindie_sim.prediction.capacity import Memory, MemoryPolicy
    from mindie_sim.prediction.structure import parse_case
    from test_prediction_system import case

    raw = case()
    memory_cfg = {"derive_workspace": False}
    memory_cfg.update(ratios or {})
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0,
                          memory=memory_cfg)
    parsed = parse_case(raw)
    base = Memory(parsed)
    capacity = int(base.weights[0] + extra)
    tight = replace(parsed, machine=replace(
        parsed.machine,
        dies=tuple(replace(die, memory_bytes=capacity)
                   for die in parsed.machine.dies)))
    return parsed, Memory(tight, MemoryPolicy(derive_workspace=False))


class _StageRecipe:
    native = False
    algorithm = None

    def __init__(self, builder):
        self._builder = builder

    def lower(self, context):
        return self._builder(context)


class _StageBook:
    def __init__(self, builder):
        self._recipe = _StageRecipe(builder)

    def expand(self, ops):
        return tuple(ops)

    def prepare(self, op, geom):
        return op, self._recipe


def _normal_execution(parsed, memory, builder, *, trace_limit=256):
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.engine import Engine, Resource

    engine = Engine([Resource("host", 1.0, unit="ms", policy="fifo")],
                    pools={"hbm": 1e9, "memory@r0:s0": 1e9},
                    pool_units={"hbm": 1, "memory@r0:s0": 1},
                    trace_limit=trace_limit)
    return Execution(parsed, CostModel(parsed), recipes=_StageBook(builder),
                     engine=engine, include_submission=False, memory=memory)


def _plan(iteration=0):
    from mindie_sim.schema.workload import IterationPlan, RequestView
    return IterationPlan((RequestView("r", 1, 4, True, "decode"),),
                         iteration=iteration)


def _work(op, iteration=0):
    from mindie_sim.schema.operators import IterationWork
    return IterationWork((op,), _plan(iteration), 1, 1, 1, None, "body",
                         "test")


def _host_op(name, *, deps=(), metadata=()):
    from mindie_sim.schema.operators import OpSpec
    return OpSpec(name, "test", "prefill", 0, 0, ("r",), (1,), (1,), (1,),
                  (1,), deps=deps, metadata=tuple(metadata))


def test_normal_add_per_item_memory_grant_and_atomic_overrun():
    from mindie_sim.prediction.engine import Allocation
    from mindie_sim.prediction.program import KernelProgram, Stage

    def builder(context):
        host = context.resource_names["host"]
        k = Allocation("k", "hbm", 4096)
        v = Allocation("v", "hbm", 4096)
        return KernelProgram((Stage("work", {host: 1.0},
                                    acquire=(k, v), free=(k, v)),))

    parsed, memory = _normal_case_with_memory(8192)
    execution = _normal_execution(parsed, memory, builder)
    step = execution.add(
        _work(_host_op("o")), 0, 0,
        memory_bindings={"o": {
            "k": {"owner": "layer:k", "storage": "kv/key",
                  "ranges": ((0, 4096),), "generation": 3, "rank": 1,
                  "stage": 0, "host_bytes": 64, "unit": "bytes",
                  "physical_key": "kv_pool"},
            "v": {"owner": "layer:v", "storage": "kv/value",
                  "ranges": ((4096, 8192),), "generation": 4, "rank": 2,
                  "stage": 0, "host_bytes": 32, "unit": "bytes",
                  "physical_key": "kv_pool"},
        }})
    execution.engine.advance(horizon=0.5, first=False)
    records = memory.grant_records
    assert set(records) == {(0, "r0:i0:o:k"), (0, "r0:i0:o:v")}
    key = records[(0, "r0:i0:o:k")]
    value = records[(0, "r0:i0:o:v")]
    assert (key["owner"], key["storage"], key["ranges"], key["generation"],
            key["rank"], key["stage"], key["host_bytes"]) == (
                "layer:k", "kv/key", ((0, 4096),), 3, 1, 0, 64.0)
    assert (value["owner"], value["storage"], value["ranges"],
            value["generation"], value["rank"], value["stage"],
            value["host_bytes"]) == (
                "layer:v", "kv/value", ((4096, 8192),), 4, 2, 0, 32.0)
    grant = execution.engine.memory_grant("r0:i0:o:k")
    assert grant["binding"]["unit"] == "bytes"
    assert memory.peak_combined[0][0] == 8192.0
    execution.engine.advance(until=(step.completion,), first=False)
    assert memory.live_grants(0, 0) == 0.0

    # One byte over: the two-item transaction is atomic, no first-item leak.
    def over_builder(context):
        host = context.resource_names["host"]
        k = Allocation("k", "hbm", 4096)
        v = Allocation("v", "hbm", 8192)
        return KernelProgram((Stage("work", {host: 1.0},
                                    acquire=(k, v), free=(k, v)),))

    parsed_tight, tight = _normal_case_with_memory(8191)
    over = _normal_execution(parsed_tight, tight, over_builder)
    blocked = over.add(
        _work(_host_op("o")), 0, 0,
        memory_bindings={"o": {
            "k": {"owner": "k", "storage": "kv/key", "generation": 3,
                  "physical_key": "kv_pool"},
            "v": {"owner": "v", "storage": "kv/value", "generation": 4,
                  "physical_key": "kv_pool"},
        }})
    over.engine.advance(horizon=2.0, first=False)
    assert tight.grant_records == {}
    assert tight.live_grants(0, 0) == 0.0
    assert blocked.completion.time is None


def test_normal_add_grant_is_held_until_real_last_use_not_acquire_completion():
    from mindie_sim.prediction.engine import Allocation
    from mindie_sim.prediction.program import KernelProgram, Stage

    def builder(context):
        host = context.resource_names["host"]
        block = Allocation("kv", "hbm", 4096)
        return KernelProgram((Stage("work", {host: 1.0},
                                    acquire=(block,), free=(block,)),))

    parsed, memory = _normal_case_with_memory(4096)
    execution = _normal_execution(parsed, memory, builder)
    reader_done = execution.engine.submit("reader_done", latency=5.0)
    first = execution.add(
        _work(_host_op("a")), 0, 0,
        memory_bindings={"a": {"kv": {
            "owner": "layer:0", "storage": "kv/page/0",
            "ranges": ((0, 4096),), "generation": 3, "rank": 0,
            "stage": 0, "last_use": reader_done}}})
    execution.engine.advance(horizon=2.0, first=False)
    # The logical free completed at 1 ms; the physical grant is still held by
    # the actual asynchronous reader until 5 ms.
    assert first.completion.time == 1.0
    assert first.device_complete.started is None or first.device_complete.done
    assert memory.live_grants(0, 0) == 4096.0
    second = execution.add(
        _work(_host_op("b"), iteration=1), 0, 2.0,
        memory_bindings={"b": {"kv": {
            "owner": "layer:1", "storage": "kv/page/1",
            "ranges": ((4096, 8192),), "generation": 4}}})
    execution.engine.advance(until=(second.completion,), first=False)
    rows = {dict(row)["task"]: row for row in execution.engine.trace}
    assert rows["r0:i0:a:work"]["start_ms"] == 0.0
    assert rows["r0:i1:b:work"]["start_ms"] == 5.0
    assert second.completion.time == 6.0
    assert memory.peak_combined[0][0] == 4096.0
    assert memory.live_grants(0, 0) == 0.0


def test_normal_add_per_reader_offload_and_named_target_draft_writeback():
    from mindie_sim.prediction.program import KernelProgram, Stage
    from mindie_sim.schema.operators import OpSpec

    def builder(context):
        host = context.resource_names["host"]
        return KernelProgram((Stage("work", {host: 1.0}),))

    parsed, memory = _normal_case_with_memory(
        0, ratios={"kvc_ddr_ratio": 0.5, "kvc_ssd_ratio": 0.25})
    execution = _normal_execution(parsed, memory, builder)
    source3 = execution.engine.submit("source3", latency=3.0)
    grant5 = execution.engine.submit("grant5", latency=5.0)
    reader = {"occurrence": "reader", "producer": "producer:0",
              "request": "r", "direction": "read", "scope": "target",
              "output": "cache_key", "storage": "kv/key", "bytes": 0.0,
              "reader": "attn", "source_ready": source3,
              "grant_ready": grant5}
    target = {"occurrence": "store", "producer": "store", "request": "r",
              "direction": "write", "scope": "target", "output": "cache_key",
              "storage": "kv/key", "bytes": 256.0}
    draft = {"occurrence": "draft", "producer": "draft", "request": "r",
             "direction": "write", "scope": "draft",
             "output": "cache_value", "storage": "kv/value", "bytes": 192.0}
    ops = (
        OpSpec("reader", "attention", "prefill", 0, 0, ("r",), (1,), (1,),
               (1,), (1,), metadata=(("state_accesses", (reader,)),)),
        OpSpec("store", "cache_store", "prefill", 0, 0, ("r",), (1,), (1,),
               (1,), (1,), deps=("reader",),
               metadata=(("state_accesses", (target,)),)),
        OpSpec("draft", "cache_store", "prefill", 0, 0, ("r",), (1,), (1,),
               (1,), (1,), deps=("store",),
               metadata=(("state_accesses", (draft,)),
                         ("producer_scope", "draft"))),
        OpSpec("proposal", "token_commit", "prefill", 0, 0, ("r",), (1,),
               (1,), (1,), (1,), deps=("draft",)),
    )
    from mindie_sim.schema.operators import IterationWork
    step = execution.add(
        IterationWork(ops, _plan(), 1, 1, 1, None, "body", "test"), 0, 0)
    execution.engine.advance(until=(step.completion,), first=False)
    serviced = execution.engine.serviced
    assert serviced["offload_ddr@r0:s0"] == pytest.approx(224.0)
    assert serviced["offload_ssd@r0:s0"] == pytest.approx(112.0)
    assert serviced["memory@r0:s0"] == pytest.approx(336.0)
    rows = {dict(row)["task"]: row for row in execution.engine.trace}
    assert step.offload_release is not None
    # target/draft writebacks use their own named producer completion.
    assert rows["r0:i0:s0.l0:kv_writeback_ddr"]["start_ms"] >= 1.0
    assert rows["r0:i0:s0:draft_kv_writeback_ddr"]["start_ms"] >= 2.0
    assert rows["r0:i0:s0.l0:kv_writeback_ddr"]["start_ms"] < 3.0
    assert step.offload_release is not None
    assert not execution._offload_access_batches


def test_normal_add_explicit_empty_accesses_are_true_zero():
    from mindie_sim.prediction.program import KernelProgram, Stage

    def builder(context):
        return KernelProgram((Stage("work", {context.resource_names["host"]: 1.0}),))

    parsed, memory = _normal_case_with_memory(
        0, ratios={"kvc_ddr_ratio": 0.5, "kvc_ssd_ratio": 0.25})
    execution = _normal_execution(parsed, memory, builder)
    op = _host_op("empty", metadata=(("state_accesses", ()),))
    step = execution.add(_work(op), 0, 0)
    execution.engine.advance(until=(step.completion,), first=False)
    assert execution.engine.serviced["offload_ddr@r0:s0"] == 0.0
    assert execution.engine.serviced["offload_ssd@r0:s0"] == 0.0
    assert execution.engine.serviced["memory@r0:s0"] == 0.0


def test_normal_add_target_tail_without_named_writer_is_not_dropped():
    from mindie_sim.prediction.program import KernelProgram, Stage
    from mindie_sim.schema.operators import OpSpec

    def builder(context):
        return KernelProgram((Stage("work", {context.resource_names["host"]: 1.0}),))

    parsed, memory = _normal_case_with_memory(
        0, ratios={"kvc_ddr_ratio": 0.5, "kvc_ssd_ratio": 0.25})
    execution = _normal_execution(parsed, memory, builder)
    tail = {"occurrence": "tail_store", "producer": "writer_not_submitted",
            "request": "r", "direction": "write", "scope": "target",
            "output": "cache_key", "storage": "kv/key", "bytes": 256.0}
    op = OpSpec("tail_store", "cache_store", "prefill", 0, 0, ("r",), (1,),
                (1,), (1,), (1,),
                metadata=(("state_accesses", (tail,)),))
    step = execution.add(_work(op), 0, 0)
    execution.engine.advance(until=(step.completion,), first=False)
    assert execution.engine.serviced["offload_ddr@r0:s0"] == pytest.approx(128.0)
    assert execution.engine.serviced["offload_ssd@r0:s0"] == pytest.approx(64.0)
    assert execution.engine.serviced["memory@r0:s0"] == pytest.approx(192.0)


def test_normal_add_per_reader_mandatory_edges_are_a_union_not_a_barrier():
    from mindie_sim.prediction.program import KernelProgram, Stage
    from mindie_sim.schema.operators import IterationWork, OpSpec

    def builder(context):
        return KernelProgram((Stage("work", {context.resource_names["host"]: 1.0}),))

    parsed, memory = _normal_case_with_memory(
        0, ratios={"kvc_ddr_ratio": 0.5, "kvc_ssd_ratio": 0.25})
    execution = _normal_execution(parsed, memory, builder)
    source3 = execution.engine.submit("source3", latency=3.0)
    grant5 = execution.engine.submit("grant5", latency=5.0)
    grant3 = execution.engine.submit("grant3", latency=3.0)
    source5 = execution.engine.submit("source5", latency=5.0)

    def reader(name, source_ready, grant_ready):
        return {"occurrence": name, "producer": name, "request": "r",
                "direction": "read", "scope": "target",
                "output": "cache_key", "storage": "kv/key", "bytes": 32.0,
                "reader": name, "source_ready": source_ready,
                "grant_ready": grant_ready}

    ops = (
        OpSpec("reader0", "attention", "prefill", 0, 0, ("r",), (1,), (1,),
               (1,), (1,),
               metadata=(("state_accesses", (reader("r0", source3, grant5),)),)),
        OpSpec("reader1", "attention", "prefill", 0, 1, ("r",), (1,), (1,),
               (1,), (1,),
               metadata=(("state_accesses", (reader("r1", source5, grant3),)),)),
    )
    step = execution.add(
        IterationWork(ops, _plan(), 1, 1, 1, None, "body", "test"), 0, 0)
    execution.engine.advance(until=(step.completion,), first=False)
    rows = [dict(row) for row in execution.engine.trace]
    transfers = [row for row in rows
                 if row["task"].endswith(":kv_offload_ddr")
                 and ":s0.l0:kv_offload_ddr" in row["task"]
                 or row["task"].endswith(":kv_offload_ddr")
                 and ":s0.l1:kv_offload_ddr" in row["task"]]
    assert len(transfers) == 2
    assert all(row["start_ms"] >= 5.0 for row in transfers)
    works = {row["task"]: row for row in rows if row["task"].endswith(":work")}
    assert works["r0:i0:reader0:work"]["start_ms"] >= 5.0
    assert works["r0:i0:reader1:work"]["start_ms"] >= 5.0


def test_12kib_rollback_waits_for_correct_last_reader_and_ignores_wrong_generation():
    from mindie_sim.prediction.engine import Allocation
    from mindie_sim.prediction.program import KernelProgram, Stage

    def builder(context):
        block = Allocation("kv", "hbm", 4096)
        return KernelProgram((Stage("work", acquire=(block,), free=(block,)),))

    probe_parsed, probe = _normal_case_with_memory(4096)
    assert probe.resize(0, "L0", 1)
    request_live = sum(probe.live[0])
    parsed, memory = _normal_case_with_memory(
        12288 + int(request_live) + 1)
    execution = _normal_execution(parsed, memory, builder)
    done5 = execution.engine.submit("reader5_done", latency=5.0)
    done6 = execution.engine.submit("reader6_done", latency=6.0)
    assert memory.resize(0, "L0", 1)
    assert memory.hold_for_async_consumer(
        0, "L0", "reader:l0", storage="kv/page/l0", generation=3,
        event=done5)
    memory.release(0, "L0")                 # logical rollback/retirement
    l0 = execution.add(
        _work(_host_op("l0")), 0, 0,
        memory_bindings={"l0": {"kv": {
            "owner": "layer:0", "storage": "kv/page/l0",
            "ranges": ((0, 4096),), "generation": 3, "last_use": done5}}})
    l1 = execution.add(
        _work(_host_op("l1"), iteration=1), 0, 0,
        memory_bindings={"l1": {"kv": {
            "owner": "layer:1", "storage": "kv/page/l1",
            "ranges": ((4096, 8192),), "generation": 4, "last_use": done6}}})
    l2 = execution.add(
        _work(_host_op("l2"), iteration=2), 0, 0,
        memory_bindings={"l2": {"kv": {
            "owner": "layer:2", "storage": "kv/page/l2",
            "ranges": ((8192, 12288),), "generation": 5, "last_use": done6}}})
    execution.engine.advance(horizon=2.0, first=False)
    assert memory.live_grants(0, 0) == 12288.0
    assert memory.is_retired(0, "L0")
    assert not memory.release_grant(0, "r0:i0:l0:kv", generation=99)
    assert memory.live_grants(0, 0) == 12288.0
    fourth = execution.add(
        _work(_host_op("l3"), iteration=3), 0, 2.0,
        memory_bindings={"l3": {"kv": {
            "owner": "layer:3", "storage": "kv/page/l3",
            "ranges": ((12288, 16384),), "generation": 6}}})
    execution.engine.advance(horizon=4.0, first=False)
    assert memory.live_grants(0, 0) == 12288.0
    assert fourth.completion.time is None
    execution.engine.advance(until=(fourth.completion,), first=False)
    rows = {dict(row)["task"]: row for row in execution.engine.trace}
    assert rows["r0:i3:l3:work"]["start_ms"] == 5.0
    assert fourth.completion.time == 5.0
    assert memory.release_async_consumer(0, "L0", "reader:l0")
    assert memory.peak_combined[0][0] >= 12288.0
    assert memory.live_grants(0, 0) == 8192.0
    execution.engine.advance(until=(done6,), first=False)
    assert memory.live_grants(0, 0) == 0.0
    assert memory.pending_async_consumers(0, "L0") == ()
    assert (l0.completion.time, l1.completion.time, l2.completion.time) == (
        0.0, 0.0, 0.0)


def test_active_frontier_materializes_only_live_ops_and_preserves_real_work():
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.engine import Engine, Resource
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.program import KernelProgram, Stage
    from mindie_sim.schema.operators import IterationWork, OpSpec

    parsed, memory = _normal_case_with_memory(0)

    class Recipe:
        native = False
        algorithm = None

        def lower(self, context):
            resource = context.resource_names["cube"]
            return KernelProgram((Stage("work", {
                resource: context.rates[resource]}),))

    class Book:
        def expand(self, ops):
            return tuple(ops)

        def prepare(self, op, geom):
            return op, Recipe()

    def run(lazy):
        engine = Engine(
            [Resource("host", 1.0, unit="ms", policy="fifo")],
            pools={"hbm": 1e9, "memory@r0:s0": 1e9},
            pool_units={"hbm": 1, "memory@r0:s0": 1})
        execution = Execution(
            parsed, CostModel(parsed), recipes=Book(), engine=engine,
            include_submission=True, lazy_frontier=lazy, host_lookahead=4)
        plan = _plan()
        ops = tuple(
            OpSpec(f"o{index}", "test", "prefill", 0, 0, ("r",), (1,), (1,),
                   (1,), (1,),
                   deps=((f"o{index - 1}",) if index else ()))
            for index in range(600))
        step = execution.add(
            IterationWork(ops, plan, 1, 1, 1, None, "body", "test"), 0, 0)
        engine.advance(until=(step.completion,), first=False)
        return engine, step

    lazy_engine, lazy_step = run(True)
    eager_engine, eager_step = run(False)
    assert lazy_step.completion.time == eager_step.completion.time == 600.0
    assert lazy_engine.serviced["matrix@r0:s0"] == 600.0
    assert eager_engine.serviced["matrix@r0:s0"] == 600.0
    # Live state follows the active dependency frontier rather than the
    # dormant 600-op chain. The eager path keeps its original exact tasks.
    assert lazy_engine.peak_tasks < 64
    assert eager_engine.peak_tasks > 500
    assert lazy_engine.summary()["live_tasks"] == 0


def test_shared_memory_change_invalidates_periodic_frontier_and_real_work_resumes():
    from mindie_sim.prediction.engine import Allocation, Engine, Resource
    from mindie_sim.prediction.program import (
        PipelinePhase, PipelineTemplate, Stage,
    )
    from mindie_sim.prediction.scheduling import start_pipeline_template

    parsed, memory = _normal_case_with_memory(4096)
    engine = Engine([Resource("r", 1.0)], pools={"hbm": 4096},
                    pool_units={"hbm": 1})
    engine.memory = memory
    engine.memory_pools = {"hbm"}
    load = Stage("load", {"r": 1.0}, previous=("load",))
    compute = Stage(
        "compute", {"r": 1.0}, after=("load",), previous=("compute",),
        acquire=(Allocation("b", "hbm", 4096),),
        free=(Allocation("b", "hbm", 4096),))
    template = PipelineTemplate(
        0, 1, (PipelinePhase("body", (load, compute), 100),))
    done = start_pipeline_template(
        engine, template, "p", engine.submit("ready"))
    engine.submit("clock", latency=10.0)
    engine.advance(horizon=4.0, first=False)
    assert engine.serviced["r"] == 4.0
    assert engine.allocations == {}
    # A real shared-budget change at the settled boundary. The next compute
    # cannot be admitted and the periodic proof must not step over it.
    assert memory.grant_batch(0, [("external", 4096)], stage=0)
    engine.advance(horizon=10.0, first=False)
    assert engine.serviced["r"] == 5.0
    assert not done.done
    assert memory.live_grants(0, 0) == 4096.0
    assert memory.release_grant(0, "external")
    engine.advance(until=(done,), first=False)
    assert engine.serviced["r"] == 200.0
    assert done.time == 205.0
    assert memory.live_grants(0, 0) == 0.0


def test_conductor_normal_submit_uses_per_allocation_memory_binding():
    from test_prediction_system import case
    parsed, memory = _normal_case_with_memory(8192)
    execution = _normal_execution(
        parsed, memory,
        lambda context: (_ for _ in ()).throw(AssertionError("unused")))
    fragment = _fragment(
        "kv", (("host", 1.0, "ms"),),
        allocations=(
            AllocationEffect.acquire("k", "hbm", 4096),
            AllocationEffect.acquire("v", "hbm", 4096),
            AllocationEffect.release("k", "hbm", 4096),
            AllocationEffect.release("v", "hbm", 4096),
        ))
    conductor = execution.fragment_conductor(
        replica=0, stage=0, memory=memory,
        memory_owner="default-owner", memory_storage="kv/default",
        memory_ranges=((0, 1),), memory_generation=1,
        memory_bindings={
            "k": {"owner": "layer:k", "storage": "kv/key",
                  "ranges": ((0, 4096),), "generation": 3, "rank": 1,
                  "host_bytes": 64, "unit": "bytes"},
            "v": {"owner": "layer:v", "storage": "kv/value",
                  "ranges": ((4096, 8192),), "generation": 4, "rank": 2,
                  "host_bytes": 32, "unit": "bytes"},
        })
    handle = conductor.submit(fragment, owner="test")
    execution.engine.advance(horizon=0.5, first=False)
    records = memory.grant_records
    assert set(records) == {(0, "fragment@0:s0:kv#1:k"),
                            (0, "fragment@0:s0:kv#1:v")}
    key = records[(0, "fragment@0:s0:kv#1:k")]
    value = records[(0, "fragment@0:s0:kv#1:v")]
    assert (key["owner"], key["storage"], key["ranges"], key["generation"],
            key["rank"], key["host_bytes"]) == (
                "layer:k", "kv/key", ((0, 4096),), 3, 1, 64.0)
    assert (value["owner"], value["storage"], value["ranges"],
            value["generation"], value["rank"], value["host_bytes"]) == (
                "layer:v", "kv/value", ((4096, 8192),), 4, 2, 32.0)
    conductor.advance(until=(handle.event,))
    assert memory.live_grants(0, 0) == 0.0


def test_second_conductor_does_not_overwrite_first_pool_or_ledger():
    parsed, first_memory = _normal_case_with_memory(4096)
    execution = _normal_execution(
        parsed, first_memory,
        lambda context: (_ for _ in ()).throw(AssertionError("unused")))
    first = execution.fragment_conductor(replica=0, stage=0,
                                        memory=first_memory)
    parsed2, second_memory = _normal_case_with_memory(8192)
    second = execution.fragment_conductor(replica=0, stage=0,
                                          memory=second_memory)
    fragment = _fragment(
        "ledger", (("host", 1.0, "ms"),),
        allocations=(AllocationEffect.acquire("b", "hbm", 4096),
                     AllocationEffect.release("b", "hbm", 4096)))
    handle = second.submit(fragment)
    execution.engine.advance(horizon=0.5, first=False)
    assert second_memory.grant_records
    assert first_memory.grant_records == {}
    assert execution.memory is first_memory
    assert second.memory is second_memory
    second.advance(until=(handle.event,))
