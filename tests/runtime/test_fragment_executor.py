"""Behavioral laws of the shared fragment executor and execution bridge.

[Role] Exercise resource arbitration, memory grants, state, ports and frontiers.
[Invariants] Preserve work and completion ordering, admit effects atomically,
  and bound live work by the active frontier.
[Timeouts & Locks] Finite in-memory simulations and bounded progress checks.
[Side-Effect Policy] No device access, filesystem writes, or environment changes.
"""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mindie_sim.observations.resource_services import normalize_resource_service
from mindie_sim.prediction.capacity import Memory, MemoryPolicy
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.engine import Allocation, Engine, ExecutionError, Resource
from mindie_sim.prediction.engine_periodic import CycleCorrespondence, CycleSkip, IntervalDelta, RunCorrespondence
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.program.primitives import PrimitiveTiming
from mindie_sim.prediction.program import KernelProgram, PipelinePhase, PipelineTemplate, Stage
from mindie_sim.prediction.scheduling import (FragmentConductor, FragmentContractError,
                                              UnsupportedFragmentSemantics, start_pipeline_template)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.invocation import Occurrence
from mindie_sim.schema.operators import (AllocationEffect, IterationWork, ProgramFragment,
                                                ResourcePolicy, StateEffect, WorkItem, WorkSet)
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.schema.subgraphs import EventPortSpec, PortSet

from _support.kernel_invocation import op
from _support.memory_cases import budget_memory, case

pytestmark = pytest.mark.tier0

_PAGE = 4 * 1024
_H = (("H", 1, "u"),)
_HBM = (("hbm", 1_000_000, "bytes"),)

def _fragment(name, work=_H, **effects):
    return ProgramFragment(name, work=WorkSet(tuple(WorkItem(*item) for item in work)), **effects)

def _slot(name="buf", pool="P", size=1, *, release=True, **kwargs):
    acquire = AllocationEffect.acquire(name, pool, size, **kwargs)
    return (acquire, AllocationEffect.release(name, pool, size, **kwargs)) if release else (acquire,)

def _ports(*ports):
    return PortSet(event_in=tuple(p for p in ports if p.direction == "in"),
                   event_out=tuple(p for p in ports if p.direction == "out"))

def _output(name, kind="complete", **options):
    return EventPortSpec(name, "out", kind, **options)

def _input(name="done", **options):
    return EventPortSpec(name, "in", "ready", **options)

def _conductor(policy="fair", *, pools=None, trace_limit=0, **options):
    engine = Engine([Resource("H", 1, unit="u", policy=policy)], pools=pools, trace_limit=trace_limit)
    return FragmentConductor(engine, **options)

@pytest.mark.parametrize("policy,coupled,priority,late,a_work,b_work,expected", [
    ("fifo", False, 0, False, 1, 2, (1, 3, 4)), ("fifo", False, 1, False, 1, 2, (3, 2, 6)),
    ("fair", False, 0, False, 1, 2, (2, 3, 5)), ("fifo", False, 1, True, 2, 1, (2, 3, 5)),
    ("fifo", True, 0, True, 1, 1, (1, 2, 4)), ("fifo", True, 1, False, 1, 1, (2, 1, 5)),
])
def test_arbitration_sets_successor_readiness_and_preserves_the_fifo_owner(
    policy, coupled, priority, late, a_work, b_work, expected,
):
    engine = Engine([Resource("H", 1, unit="u", policy=policy), Resource("C", 10, unit="u"),
                     Resource("successor", 1, policy="fifo")])
    conductor = FragmentConductor(engine)
    def submit(name, amount, priority=0):
        work = (("H", amount, "u"),) + ((("C", 10, "u"),) if coupled else ())
        return conductor.submit(_fragment(name, work, coupled=coupled, priority=priority)).event
    a = submit("a", a_work, priority)
    if late:
        engine.advance(horizon=.5)
    b = submit("b", b_work)
    c = engine.submit("after_a", {"successor": 3}, after=(a,))
    engine.advance(until=(b, c), first=False)
    assert (a.time, b.time, c.time) == expected
    assert engine.serviced == {"H": a_work + b_work, "C": 20 * coupled, "successor": 3}

@pytest.mark.parametrize("coupled,repeats,busy", [(False, 1, 1), (True, 1, 3), (True, 2, 6)])
def test_coupled_service_preserves_work_busy_time_and_repeat_batching(coupled, repeats, busy):
    engine = Engine([Resource(name, 10, unit="u") for name in ("H", "C")])
    conductor = FragmentConductor(engine)
    done = conductor.submit(_fragment("both", (("H", 30, "u"), ("C", 10, "u")), coupled=coupled, repeats=repeats))
    conductor.advance()
    assert done.batched == (repeats > 1) and done.event.time == 3 * repeats
    assert engine.serviced == {"H": 30 * repeats, "C": 10 * repeats}
    assert engine.busy_ms == {"H": 3 * repeats, "C": busy}

@pytest.mark.parametrize("policy,repeats,rival_work", [("fifo", 2, 1), ("fair", 2, 1), ("fifo", 2000, 0), ("fair", 3, 0)])
def test_repeat_batching_matches_expansion_and_bounds_live_work(policy, repeats, rival_work):
    conductor = _conductor(policy)
    batch = conductor.submit(_fragment("repeated", repeats=repeats))
    rival = conductor.submit(_fragment("rival")) if rival_work else None
    conductor.advance()
    assert batch.batched == (policy == "fair") and batch.repeat_count == repeats
    assert batch.event.time == conductor.engine.serviced["H"] == repeats + rival_work
    assert conductor.engine.peak_tasks <= 2 and (rival is None or rival.event.time == 2)
    if batch.batched and rival is None:
        assert conductor.engine.completed_count == 1
        end = batch.event
        for index in range(repeats):
            end = conductor.submit(_fragment(str(index)), after=(end,)).event
        conductor.advance()
        assert end.time - batch.event.time == repeats
        with pytest.raises(FragmentContractError, match="fragment_batched_work_overflow"):
            conductor.submit(_fragment("overflow", repeats=10 ** 400))

def test_independent_multiresource_repeats_preserve_late_arrivals():
    engine = Engine([Resource(name, 1, unit="u") for name in ("H", "C")])
    conductor = FragmentConductor(engine)
    slow = conductor.submit(_fragment("slow", (*_H, ("C", 10, "u")), repeats=2))
    conductor.advance(horizon=1.1)
    late = conductor.submit(_fragment("late"))
    conductor.advance()
    assert not slow.batched and (slow.event.time, late.event.time) == pytest.approx((20, 2.1))

@pytest.mark.parametrize("scope,condition,enabled,repeats,limit,expected", [
    ("stage", "", True, 2, 1, "fragment_repeats_exceed_expansion_bound"),
    ("program", "", True, 2, 1, "fragment_repeats_exceed_expansion_bound"),
    ("event", "", True, 1, 1, "allocation_scope_unsupported:event"),
    ("stage", "use", None, 1, 1, "conditional_effect_unbound:buf:use"),
    ("stage", "use", False, 1, 1, 0), ("stage", "use", True, 1, 1, 1),
    ("stage", "", True, 2, 2, 1), ("program", "", True, 2, 2, 1),
])
def test_allocation_scope_condition_and_repeat_bound_set_residency(scope, condition, enabled, repeats, limit, expected):
    conductor = _conductor(pools={"P": 1}, repeat_limit=limit)
    engine = conductor.engine
    fragment = _fragment("resident", repeats=repeats, allocations=_slot(lifetime=scope, condition=condition))
    conditions = {"use": enabled} if enabled is not None else {}
    if isinstance(expected, str):
        error = FragmentContractError if enabled is None else UnsupportedFragmentSemantics
        with pytest.raises(error, match=expected):
            conductor.submit(fragment, conditions=conditions)
        assert not engine.summary()["live_tasks"] and not engine.allocations
    else:
        handle = conductor.submit(fragment, conditions=conditions)
        if repeats > 1:
            conductor.advance(horizon=1)
            assert engine.allocations and not handle.batched
        conductor.advance()
        assert handle.event.time == repeats and engine.peak_allocated == {"P": expected}
        assert not engine.allocations

@pytest.mark.parametrize("lifetime,delays", [("stage", (5,)), ("program", (5,)), ("program", (3, 7)), ("program", (0, 7))])
def test_allocation_releases_follow_individual_last_use(lifetime, delays):
    conductor = _conductor(pools={"P": 2})
    engine = conductor.engine
    holds = {str(i): engine.submit(str(i), latency=t) for i, t in enumerate(delays) if t}
    effects = tuple(effect for i, t in enumerate(delays)
                    for effect in _slot(str(i), release=not t, lifetime=lifetime))
    running = conductor.submit(_fragment("running", allocations=effects), holds=holds)
    blocked = conductor.submit(_fragment("blocked", allocations=_slot(size=2)), after=(engine.submit("arrive", latency=.5),))
    conductor.advance(horizon=2)
    assert running.event.time == 1 and blocked.event.started is None and engine.allocations
    conductor.advance()
    assert [event.time for _, event in running.release_by_name] == [delay or 1 for delay in delays]
    assert blocked.event.started == max(delays) and blocked.event.time == max(delays) + 1
    assert engine.busy_ms == {"H": 2} and not engine.allocations

@pytest.mark.parametrize("pools", [{"P": 2}, {"P": 1, "Q": 1}])
def test_fragment_grants_are_instance_scoped_and_admit_all_pools_atomically(pools):
    conductor = _conductor(pools=pools)
    engine = conductor.engine
    if "Q" in pools:
        free = engine.submit("free_q", latency=10)
        held = _slot("q", "Q", lifetime="program", release=False)
        blocker = conductor.submit(_fragment("blocker", (), allocations=held), holds={"q": free})
        effects = _slot("p", lifetime="program") + _slot("q", "Q", lifetime="program")
        low, high = (conductor.submit(_fragment("same", allocations=effects, priority=p)) for p in (1, 0))
        conductor.advance()
        assert blocker.release_events[0].time == 10
        for handle, start in ((high, 10), (low, 11)):
            assert (handle.event.started, handle.event.time) == (start, start + 1)
            assert [event.time for _, event in handle.grant_by_name] == [start, start]
    else:
        fragment = _fragment("same", allocations=_slot())
        first, second = (conductor.submit(fragment) for _ in range(2))
        conductor.advance()
        assert first.event.time == second.event.time == engine.peak_allocated["P"] == 2
        engine.submit("free", free=(Allocation("missing", "P", 1),))
        with pytest.raises(ExecutionError, match="release_without_live_allocation"):
            engine.advance()
    assert engine.serviced == {"H": 2} and not engine.allocations

def _state(conductor, action, version, *after, work=(), condition="", arbiter="", repeats=1):
    effect = StateEffect("kv", action, version=version, condition=condition, arbiter=arbiter)
    return conductor.submit(_fragment(action, work, state_effects=(effect,), repeats=repeats),
                            after=tuple(handle.event for handle in after), conditions={"use": False})

@pytest.mark.parametrize("action,condition,version", [("rollback", "", 7), ("commit", "use", 7), ("commit", "", 8)])
def test_state_visibility_at_same_timestamp_and_repeated_reads(action, condition, version):
    conductor = _conductor(state={"kv": 7})
    read = _state(conductor, "read", 7)
    write = _state(conductor, "write", 8, read)
    decision = _state(conductor, action, 8, write, condition=condition)
    reader = _state(conductor, "read", version, decision, work=_H, repeats=2)
    conductor.advance()
    assert not reader.batched and reader.event.started == decision.event.time == 0 and reader.event.time == 2
    assert action != "rollback" or conductor.state.live_versions("kv") == (7,)
    _state(conductor, "read", 9)
    with pytest.raises(FragmentContractError, match="state_version_unavailable"):
        conductor.advance()

@pytest.mark.parametrize("arbiters,winner", [(("", ""), None), (("a", "b"), 9), (("b", "a"), 8)])
@pytest.mark.parametrize("reverse", [False, True])
def test_simultaneous_commits_require_a_deterministic_total_order(arbiters, winner, reverse):
    conductor = _conductor(state={"kv": 7})
    writes = tuple(_state(conductor, "write", version) for version in (8, 9))
    commits = list(zip((8, 9), arbiters))
    for version, arbiter in reversed(commits) if reverse else commits:
        _state(conductor, "commit", version, *writes, arbiter=arbiter)
    if winner is None:
        with pytest.raises(FragmentContractError, match="state_effect_conflict_requires_arbiter"):
            conductor.advance()
    else:
        conductor.advance()
        assert conductor.state.committed_version("kv") == winner

def test_state_and_observation_preserve_schedule_with_bounded_history():
    results = []
    for detail in (False, True):
        conductor = _conductor(state={"kv": 0}, cache_limit=8, detail=detail, trace_limit=4)
        for version in range(1, 401):
            write = _state(conductor, "write", version, work=_H)
            commit = _state(conductor, "commit", version, write)
            repeat = conductor.submit(_fragment(f"repeat{version}", repeats=100), after=(commit.event,))
            conductor.advance(until=(repeat.event,))
        summary = conductor.summary()
        assert conductor.state.committed_version("kv") == 400 and len(conductor.state.live_versions("kv")) <= 2
        assert summary["live_fragments"] == summary["engine"]["live_tasks"] == 0
        assert len(summary["fragments"]) <= 8 and summary["aggregates_evicted"] > 0
        assert len(summary["recent"]) == (8 if detail else 0) and bool(summary.get("recent_dropped")) == detail
        results.append((repeat.event.time, summary["engine"], summary["totals"]))
    assert results[0] == results[1]

def test_stream_order_uses_recorded_event_and_optional_inputs_can_be_unbound():
    conductor = _conductor()
    with pytest.raises(FragmentContractError, match="fragment_input_unbound:missing"):
        conductor.submit(_fragment("required", ports=_ports(_input("missing"))))
    controls = (("producer", "s", _output("done")), ("consumer", "s", _input()),
                ("independent", "other", _input("missing", optional=True)))
    events = [conductor.submit(_fragment(name, ports=_ports(port),
                                         occurrence=Occurrence("stage", name, stream=stream))).event
              for name, stream, port in controls]
    conductor.advance()
    assert [(e.started, e.time) for e in events] == [(0, 2), (2, 3), (0, 2)]

@pytest.mark.parametrize("count,limit", [(2, 8), (4, 2)])
def test_event_generations_disambiguate_and_evict_at_bounded_capacity(count, limit):
    conductor = _conductor(event_limit=limit)
    for index in range(count):
        conductor.record_event("done", conductor.engine.submit(str(index), latency=index + 1))
    assert conductor.summary()["events_evicted"] == count - 2
    with pytest.raises(FragmentContractError, match="fragment_event_generation_required:done"):
        conductor.submit(_fragment("implicit", ports=_ports(_input())))
    if count > limit:
        with pytest.raises(FragmentContractError, match="fragment_input_unbound:done"):
            conductor.submit(_fragment("evicted", ports=_ports(_input(generation=0))))
    handle = conductor.submit(_fragment("bound", ports=_ports(_input(generation=count - 1))))
    conductor.advance()
    assert (handle.event.started, handle.event.time) == (count, count + 1)

def test_event_outputs_bind_grant_release_completion_and_explicit_record():
    conductor = _conductor(pools={"P": 2})
    mark = conductor.engine.submit("recorded", latency=3)
    ports = _ports(_output("buf_grant", "grant"), _output("buf_release", "release"),
                   _output("done"), _output("recorded", "record"))
    fragment = _fragment("effects", allocations=_slot(lifetime="program"), ports=ports)
    with pytest.raises(UnsupportedFragmentSemantics, match="fragment_event_port_requires_explicit_binding:record:recorded"):
        conductor.submit(fragment)
    handle = conductor.submit(fragment, event_bindings={"recorded": mark})
    conductor.advance()
    assert conductor.event("recorded") is mark and conductor.event("done") is handle.event
    assert [conductor.event(f"buf_{kind}").time for kind in ("grant", "release")] == [0, 1]

@pytest.mark.parametrize("kind,match", [
    ("names", "fragment_event_name_limit_exceeded:2"),
    ("generation", "fragment_event_generation_already_recorded:existing:0"),
    ("binding", "fragment_event_binding_invalid:event:x"),
    ("ambiguous", "fragment_event_port_ambiguous:release:x"),
])
def test_event_preflight_rejects_batch_before_engine_or_registry_changes(kind, match):
    conductor = _conductor(event_limit=2, pools={"P": 2})
    engine = conductor.engine
    existing = engine.submit("existing")
    conductor.record_event("existing", existing, generation=0)
    outputs = {"names": (_output("x"), _output("y")), "generation": (_output("existing", generation=0),),
               "binding": (_output("x", "record"),), "ambiguous": (_output("x", "release"),)}
    effects = _slot("a", lifetime="program") + _slot("b", lifetime="program") if kind == "ambiguous" else ()
    fragment = _fragment("rejected", ports=_ports(*outputs[kind]), allocations=effects)
    with pytest.raises(UnsupportedFragmentSemantics if kind == "ambiguous" else FragmentContractError, match=match):
        conductor.submit(fragment, event_bindings={"x": object()} if kind == "binding" else {})
    assert engine.summary()["live_tasks"] == 1 and not engine.allocations and engine.serviced == {"H": 0}
    assert conductor.summary()["recorded_events"] == 1 and conductor.summary()["totals"]["fragments"] == 0
    handle = conductor.submit(_fragment("accepted", ports=_ports(_output("x"))))
    conductor.advance()
    with pytest.raises(FragmentContractError, match="fragment_event_name_limit_exceeded:2"):
        conductor.record_event("y", existing)
    assert conductor.event("existing", 0) is existing and conductor.event("x") is handle.event
    with pytest.raises(FragmentContractError, match="fragment_event_unbound:y"):
        conductor.event("y")

@pytest.mark.parametrize("binding,value,match", [
    ("policy", "fair", "fragment_resource_policy_mismatch:H:H:fair:fifo"),
    ("policy", "fifo", "fragment_resource_policy_mismatch:H:H:fifo:fair"),
    ("unit", "cycles", "fragment_work_unit_mismatch:H:cycles:bytes"),
])
def test_resource_binding_requires_compatible_policy_and_units(binding, value, match):
    policy = ("fifo" if value == "fair" else "fair")
    engine = Engine([Resource("H", 8, unit="bytes", policy=policy)])
    policies = (("H", ResourcePolicy(sharing=value)),) if binding == "policy" else ()
    fragment = _fragment("work", (("H", 16, "cycles" if binding == "unit" else "bytes"),), policies=policies)
    with pytest.raises(FragmentContractError, match=match):
        FragmentConductor(engine).submit(fragment)
    if binding == "unit":
        conductor = FragmentConductor(engine, unit_conversions={("cycles", "bytes"): 8})
        handle = conductor.submit(fragment)
        conductor.advance()
        assert handle.event.time == .25 and engine.serviced == {"H": 2}

def test_engine_boundaries_and_step_budgets_preserve_causal_progress():
    engine = Engine([Resource("H", 1)], zero_progress_limit=6)
    first = engine.submit("first", {"H": 2})
    middle = engine.submit("middle", {"H": 1}, after=(first,))
    end = engine.submit("end", {"H": 1}, after=(middle,))
    started = engine.start_boundary(middle, "started")
    prepared = engine.boundary("prepared", after=(first,))
    with pytest.raises(ExecutionError, match="execution_step_budget_exhausted"):
        engine.advance(until=(end,), max_steps=2)
    engine.advance(until=(end,), max_steps=2)
    assert started.time == prepared.time == middle.started == engine.start_boundary(middle, "late").time == 2
    with pytest.raises(ExecutionError, match="start_boundary_requires_active_task"):
        engine.start_boundary(engine.future("unstarted"))
    def again(event):
        engine.submit("again", after=(event,), on_complete=again)
    engine.submit("spin", on_complete=again)
    with pytest.raises(ExecutionError, match="execution_zero_progress"):
        engine.advance()

def _operation(name="work", **changes):
    return replace(op(name), **({"requests": ("r",), "dtype": ""} | changes))

def _work(*ops, iteration=0, capture=None, requests=("r",)):
    plan = IterationPlan(tuple(RequestView(r, 1, 4, True, "decode") for r in requests), iteration=iteration)
    return IterationWork(ops, plan, 1, len(requests), len(requests), capture, "decode", "control")

def _execution(raw=None, **options):
    parsed = parse_case(raw or case())
    return Execution(parsed, CostModel(parsed), **options)

@pytest.mark.parametrize("mode", ["empty", "eager", "replay"])
def test_host_submit_boundaries_follow_real_launches(mode):
    raw = case(pp=2 if mode == "eager" else 1)
    if mode == "replay":
        raw["runtime"].update(graph_mode="auto", graph_capture_keys=[4], graph_capture_mode="full",
                              submission={"graph_replay_ms": .25})
    execution = _execution(raw)
    ops = tuple(_operation(str(i), stage=s, phase="decode", hbm_bytes=1000, dtype="bf16")
                for i, s in enumerate({"empty": (), "eager": (0, 1), "replay": (0,) * 5}[mode]))
    step = execution.add(_work(*ops, capture=4 if mode == "replay" else None), 0, 0)
    start = execution.stage_submit_boundary(0, 0, step=step)
    execution.engine.advance(until=(step.completion,))
    handles, count = [e for _, _, e in step.host_submits], {"empty": 0, "eager": 2, "replay": 1}[mode]
    assert len({e.sequence for e in handles}) == len(step.to_dict()["host_submits"]) == count
    assert [event.time for event in handles] == [0] * count
    assert dict(execution.submission_counts) == dict(eager_invocations=2 * (mode == "eager"), graph_replays=int(mode == "replay"))
    assert (start.time if start else None) == (handles[0].time if handles else None)

def test_operator_and_conductors_share_engine_state_and_commit_events():
    execution = _execution(case(pp=2), fragment_state={"kv": 7}, include_submission=False)
    assert execution.stage_submit_boundary(0, 0) is None
    work = _operation(hbm_bytes=1_000_000, dtype="bf16", requests=("r", "other"))
    commit = _operation("commit", family="token_commit", recipe="token_commit", deps=("work",))
    step = execution.add(_work(work, commit, requests=("r", "other")), 0, 0)
    first, second = (execution.fragment_conductor(replica=0, stage=s) for s in (0, 1))
    write = first.submit(_fragment("write", _HBM, state_effects=(StateEffect("kv", "write", version=8),)),
                         after=(step.device_complete,))
    visible = _state(first, "commit", 8, write)
    read = _state(second, "read", 8, visible, work=_HBM)
    cached = execution.submit_fragment(_fragment("cached", _HBM), after=(read.event,))
    execution.engine.advance(until=(cached.event,), first=False)
    assert write.event.started >= step.device_complete.time and read.event.time < cached.event.time
    assert first.state is second.state is execution.fragment_state
    assert tuple(step.declared_commits) == ("r",) and tuple(step.fallback_commits) == ("other",)

@pytest.mark.parametrize("basis", ["primitive", "response", "resource", "transport"])
def test_execution_context_binds_declared_services_and_only_observes_diagnostic_floors(basis, monkeypatch):
    execution = _execution(include_submission=False)
    op = _operation(phase="decode", hbm_bytes=1000, dtype="bf16")
    if basis == "transport":
        recipe = SimpleNamespace(algorithm="direct_peer", plan_options={"max_inflight_per_rank": 2, "chunk_bytes": 64})
        op = replace(op, family="alltoall", hbm_bytes=0, link_bytes=128)
        transport = execution._context(op, 0, "none", recipe=recipe).transport_options
        assert transport is execution.transport_options_for(op, 0, "none")
        assert (transport.max_inflight_per_rank, transport.chunk_bytes) == (2, 64)
    elif basis == "resource":
        execution.resource_services = {"svc": normalize_resource_service(dict(
            id="svc", scope="independent_resource", resource="hbm", phase="decode", work_unit="bytes",
            boundary="resource_phase", startup_us=1000, response_us=0, rate_scope="aggregate", active_units=1,
            supply={"kind": "absolute_rate", "rate": 1_000_000}, conditions={}, identification={}))}
        op = replace(op, metadata=(("resource_service", dict(service_id="svc", resource_role="hbm",
                                                             work_unit="bytes", phase="decode")),))
        context = SimpleNamespace(op=op, resource_names={"hbm": "memory"}, work_units={"hbm": "bytes"})
        engine = Engine([Resource("memory", 1000, unit="bytes")])
        work = KernelProgram((Stage("work", {"memory": 1000}),))
        done = execution._apply_resource_service_program(context, work).start(engine, "svc", engine.submit("ready"))
        engine.advance(until=(done,))
        assert done.time == 2 and engine.serviced == {"memory": 1000}
    else:
        response = SimpleNamespace(mode="declared_service", duration_ms=5,
                                   diagnostics=lambda: {"mode": "declared_service", "response_us": 5000})
        resolve = Mock(return_value=response)
        if basis == "primitive":
            monkeypatch.setattr(execution.cost, "lookup_diagnostic", lambda *_: {"timing": PrimitiveTiming(5, 1, 5, 5, ())})
        else:
            monkeypatch.setattr(execution.regimes, "resolve", resolve)
        production = execution._context(op, 0, "none")
        assert production.minimum_ms == 0 and production.work["memory@r0:s0"] == 1000
        resolve.assert_not_called()
        execution.legacy_diagnostics = True
        diagnostic = execution._context(op, 0, "none")
        assert diagnostic.minimum_ms == 5
        if basis == "response":
            assert diagnostic.service_regime["mode"] == response.mode and resolve.call_count == 1

def _stage_book(lower):
    recipe = SimpleNamespace(native=False, algorithm=None, lower=lower)
    return SimpleNamespace(expand=tuple, prepare=lambda op, _: (op, recipe))

def _memory_execution(extra, allocations=(), *, offload=False, duration=1):
    cfg = {"derive_workspace": False, "kvc_ddr_ratio": .5 * offload, "kvc_ssd_ratio": .25 * offload}
    def lower(context):
        return KernelProgram((Stage("work", {context.resource_names["host"]: duration}, acquire=allocations, free=allocations),))
    pools = dict.fromkeys(("hbm", "memory@r0:s0"), 1e9)
    resources = [Resource("host", 1, unit="ms", policy="fifo"), Resource("fair", 1, unit="u")]
    engine = Engine(resources, pools=pools, pool_units=dict.fromkeys(pools, 1), trace_limit=256)
    return _execution(case() | {"runtime": {"memory": cfg, **({"metadata": {"offload": {"ddr_bw_gbps": 200.0, "ssd_bw_gbps": 14.0}}} if offload else {})}}, recipes=_stage_book(lower), engine=engine, include_submission=False,
                      memory=budget_memory(extra, policy=MemoryPolicy(**cfg)))

@pytest.mark.parametrize("entry", ["operator", "blocked", "conductor", "second_conductor"])
def test_memory_grants_preserve_per_item_bindings_and_admit_atomically(entry):
    slots = tuple(Allocation(name, "hbm", _PAGE) for name in ("k", "v"))
    bindings = {slot.name: dict(owner=f"layer:{slot.name}", storage=f"kv/{slot.name}", generation=3 + i,
                                ranges=((i * _PAGE, (i + 1) * _PAGE),), rank=0, host_bytes=64 / (i + 1))
                for i, slot in enumerate(slots)}
    execution = _memory_execution(2 * _PAGE - (entry == "blocked"), slots)
    engine, memory = execution.engine, execution.memory
    if entry in ("operator", "blocked"):
        tagged = {name: dict(row, unit="bytes", physical_key="kv_pool") for name, row in bindings.items()}
        done = execution.add(_work(_operation()), 0, 0, memory_bindings={"work": tagged}).completion
    else:
        original = execution.fragment_conductor(replica=0, stage=0)
        memory = budget_memory(2 * _PAGE) if entry == "second_conductor" else original.memory
        conductor = execution.fragment_conductor(replica=0, stage=0, memory=memory, memory_owner="default",
                                                  memory_storage="kv/default", memory_bindings=bindings)
        effects = tuple(effect for slot in slots for effect in _slot(slot.name, "hbm", _PAGE))
        done = conductor.submit(_fragment("kv", (("host", 1, "ms"),), allocations=effects)).event
    engine.advance(horizon=.5, first=False)
    if entry == "blocked":
        assert not memory.grant_records and not done.done and memory.live_grants(0, 0) == 0
        return
    records = {key.rsplit(":", 1)[-1]: row for (_, key), row in memory.grant_records.items()}
    assert memory.peak_combined[0][0] == 2 * _PAGE
    assert {name: {k: row[k] for k in bindings[name]} for name, row in records.items()} == bindings
    assert entry != "operator" or engine.memory_grant("r0:i0:work:k")["binding"]["unit"] == "bytes"
    assert (execution.memory is memory) == (entry != "second_conductor")
    engine.advance(until=(done,))
    assert memory.live_grants(0, 0) == 0

@pytest.mark.parametrize("held", [1, 3])
def test_logical_retirement_preserves_grants_until_matching_last_reader(held):
    # A two-layer KV row uses 2 layers * 2 heads * 8 dim * 2 (K/V) * 2 bytes.
    execution = _memory_execution(held * _PAGE + 129 * (held == 3),
                                  (Allocation("kv", "hbm", _PAGE),), duration=int(held == 1))
    engine, memory = execution.engine, execution.memory
    early, late = (engine.submit(str(t), latency=t) for t in (5, 6))
    if held == 3:
        assert memory.resize(0, "request", 1)
        assert memory.hold_for_async_consumer(0, "request", "reader", storage="kv/page", generation=3, event=early)
        memory.release(0, "request")
    def submit(i, ready=0, last_use=None):
        name = f"op{i}"
        binding = dict(owner=name, storage=f"kv/page/{i}", generation=3 + i,
                       ranges=((i * _PAGE, (i + 1) * _PAGE),), last_use=last_use)
        return execution.add(_work(_operation(name), iteration=i), 0, ready, memory_bindings={name: {"kv": binding}})
    steps = [submit(i, last_use=late if i else early) for i in range(held)]
    engine.advance(horizon=2, first=False)
    assert [s.completion.time for s in steps] == [int(held == 1)] * held
    assert not memory.release_grant(0, "r0:i0:op0:kv", generation=99) and memory.live_grants(0, 0) == held * _PAGE
    waiting = submit(held, ready=2)
    engine.advance(horizon=4)
    assert not waiting.completion.done
    engine.advance(until=(waiting.completion,))
    assert waiting.completion.time == 5 + int(held == 1)
    if held == 3:
        assert memory.is_retired(0, "request") and memory.release_async_consumer(0, "request", "reader")
    engine.advance(until=(late,))
    assert memory.live_grants(0, 0) == 0 and not memory.pending_async_consumers(0, "request")

@pytest.mark.parametrize("mode,physical", [("empty", 0), ("unnamed", 256), ("writeback", 448), ("readers", 64)])
def test_offload_obeys_per_reader_readiness_and_producer_owned_tails(mode, physical):
    execution = _memory_execution(0, offload=True)
    engine = execution.engine
    def access(name, direction, size, **options):
        layer, deps = options.pop("layer", 0), options.pop("deps", ())
        row = dict(occurrence=name, producer=name, request="r", direction=direction, scope="target",
                   output="cache_key", storage="kv/key", bytes=size) | options
        family = "attention" if direction == "read" else "cache_store"
        return _operation(name, family=family, layer=layer, deps=deps,
                          metadata=(("state_accesses", (row,)), ("producer_scope", row["scope"])))
    ops = {"empty": (_operation(metadata=(("state_accesses", ()),)),),
           "unnamed": (access("store", "write", 256, producer="absent"),)}.get(mode)
    if ops is None:
        early, late = (engine.submit(str(t), latency=t) for t in (3, 5))
        readers = mode == "readers"
        read = access("reader", "read", 32 * readers, reader="reader", source_ready=early, grant_ready=late)
        ops = (read, access("second", "read", 32, layer=1, reader="second", source_ready=late, grant_ready=early)) if readers else (
            read, access("store", "write", 256, deps=("reader",)),
            access("draft", "write", 192, scope="draft", deps=("store",), storage="kv/value", output="cache_value"),
            _operation("proposal", family="token_commit", deps=("draft",)))
    step = execution.add(_work(*ops), 0, 0)
    engine.advance(until=(step.completion,), first=False)
    ddr_role, idle_role = ("offload_ddr_h2d", "offload_ddr_d2h") if mode == "readers" else ("offload_ddr_d2h", "offload_ddr_h2d")
    assert engine.serviced[f"{idle_role}@r0:s0"] == 0.0
    assert tuple(engine.serviced[f"{role}@r0:s0"] for role in (ddr_role, "offload_ssd", "memory")) == pytest.approx(
        (physical / 2, physical / 4, physical * .75))
    starts = {r["task"]: r["start_ms"] for r in engine.trace}
    if mode == "readers":
        waits = [start for task, start in starts.items() if task.endswith(":kv_offload_ddr")]
        assert len(waits) == 2 and min(waits) >= 5
    if mode == "writeback":
        assert 1 <= starts["r0:i0:s0.l0:kv_writeback_ddr"] < 3
        assert starts["r0:i0:s0:draft_kv_writeback_ddr"] >= 2 and step.offload_release is not None

def test_lazy_execution_frontier_matches_eager_without_materializing_dormant_work():
    def lower(context):
        cube = context.resource_names["cube"]
        return KernelProgram((Stage("work", {cube: context.rates[cube]}),))
    ops = tuple(_operation(f"op{i}", deps=(f"op{i - 1}",) if i else ()) for i in range(600))
    for lazy, limit in ((False, 500), (True, 64)):
        execution = _execution(recipes=_stage_book(lower), lazy_frontier=lazy, host_lookahead=4)
        step = execution.add(_work(*ops), 0, 0)
        engine = execution.engine
        engine.advance(until=(step.completion,))
        assert step.completion.time == engine.serviced["matrix@r0:s0"] == len(ops)
        assert (engine.peak_tasks < limit) == lazy and engine.summary()["live_tasks"] == 0

def test_shared_memory_change_invalidates_pipeline_periodicity_and_resumes_work():
    engine = Engine([Resource("r", 1)], pools={"hbm": _PAGE}, pool_units={"hbm": 1})
    engine.memory, engine.memory_pools = budget_memory(_PAGE), {"hbm"}
    scratch = Allocation("buffer", "hbm", _PAGE)
    load = Stage("load", {"r": 1}, previous=("load",))
    compute = Stage("compute", {"r": 1}, after=("load",), previous=("compute",), acquire=(scratch,), free=(scratch,))
    template = PipelineTemplate(0, 1, (PipelinePhase("body", (load, compute), 100),))
    done = start_pipeline_template(engine, template, "p", engine.submit("ready"))
    engine.submit("clock", latency=10)
    engine.advance(horizon=4, first=False)
    assert engine.memory.grant_batch(0, [("external", _PAGE)], stage=0)
    engine.advance(horizon=10, first=False)
    assert engine.serviced["r"] == 5 and not done.done
    assert engine.memory.release_grant(0, "external")
    engine.advance(until=(done,), first=False)
    assert engine.serviced["r"] == 200 and done.time == 205 and engine.memory.live_grants(0, 0) == 0

def test_periodic_skip_conserves_frontier_and_retires_real_tail_once():
    engine = Engine([Resource("r", 1)], pools={"p": 100}, pool_units={"p": 1})
    phases = tuple(PipelinePhase(name, (Stage(name, {"r": 1}),), count)
                   for name, count in (("body", 67), ("tail", 1)))
    done = start_pipeline_template(engine, PipelineTemplate(0, 2, phases), "m", engine.submit("ready"))
    engine.advance(until=(engine.submit("stop", latency=2),))
    run_id, run = next(iter(engine._periodic_runs.items()))
    interval = IntervalDelta(1, {run_id: (1, 1)}, 4, {"r": 1}, {"r": 1}, 2, {"p": 8}, 1, False)
    match = RunCorrespondence(run_id, 0, 67, 4, 67, 2, 65, 63, 63)
    sequence, allocation_events = engine._sequence, engine.allocation_event_count
    engine._apply_periodic_skip(CycleSkip(63, 1, {run_id: 1}, 4, interval, CycleCorrespondence((match,), (), 252, {run_id: 63})))
    assert engine.now == engine.serviced["r"] == engine.busy_ms["r"] == 65 and engine.peak_allocated["p"] == 8
    assert (engine._sequence - sequence, engine.allocation_event_count - allocation_events) == (252, 126)
    assert (run["next"], run["finished"], run["live_tiles"], run["slots"]) == (67, 65, {65, 66}, (0, 1))
    engine.advance(until=(done,), max_steps=100)
    assert done.time == engine.serviced["r"] == 68 and not engine._periodic_runs and not engine._periodic_tasks

def test_conductor_shares_memory_budget_and_async_reader_generation():
    execution = _memory_execution(3 * _PAGE)
    engine, memory = execution.engine, execution.memory
    conductor = execution.fragment_conductor(replica=0, stage=0)
    fragment = _fragment("resident", (("fair", 1, "u"),), allocations=_slot("a", "hbm", _PAGE))
    ends = tuple(conductor.submit(fragment, owner="request").event for _ in range(4))
    engine.advance(until=ends, first=False, max_steps=100)
    assert [event.time for event in ends] == pytest.approx([3, 3, 3, 4])
    assert memory.live_grants(0, 0) == 0 and memory.peak_combined[0][0] == 3 * _PAGE
    memory = Memory(parse_case(case()), MemoryPolicy(derive_workspace=False))
    conductor = _execution(memory=memory, include_submission=False).fragment_conductor(replica=0, stage=0)
    assert memory.resize(0, "request", 1)
    assert conductor.hold_async_reader("request", "reader", storage="kv/page", ranges=(0, 32), generation=3)
    assert memory.pending_async_consumers(0, "request") == ("reader",)
    assert conductor.release_async_reader("request", "reader") and not memory.pending_async_consumers(0, "request")


def test_tensor_input_port_enforces_producer_completion_and_causal_commits_supersede():
    from mindie_sim.prediction.runtime.contracts import PortBinding
    from mindie_sim.schema.subgraphs import TensorPortSpec

    port = TensorPortSpec("x", "in", "activation", dtype="bf16", readiness="producer_complete")
    reader = ProgramFragment(
        "read_x", ports=PortSet(tensor_in=(port,)),
        work=WorkSet((WorkItem("cube", 1, "work"),)),
    )
    for use_binding in (False, True):
        engine = Engine([Resource("cube", 1)], backend="python")
        conductor = FragmentConductor(engine)
        producer = engine.submit("produce_x", latency=4)
        port_input = PortBinding(ready=producer) if use_binding else producer
        done = conductor.submit(reader, inputs={"x": port_input}).event
        engine.advance(until=(producer, done), first=False)
        assert (producer.time, done.started, done.time) == pytest.approx((4.0, 4.0, 5.0))

    # Direct causal chain and transitive Engine boundary at t=0 both supersede prior commit.
    for via_boundary in (False, True):
        state_engine = Engine([], backend="python")
        state_conductor = FragmentConductor(state_engine, state={"kv": 0})
        def commit(version, arbiter, after=()):
            effects = (
                StateEffect("kv", "read", version - 1),
                StateEffect("kv", "write", version),
                StateEffect("kv", "commit", version, arbiter=arbiter),
            )
            return state_conductor.submit(ProgramFragment(str(version), state_effects=effects), after=after).event

        first = commit(1, "z")
        dep = state_engine.boundary("sync", after=(first,)) if via_boundary else first
        second = commit(2, "a", after=(dep,))
        state_conductor.advance(until=(second,))
        assert (first.time, second.time, state_conductor.state.committed_version("kv")) == (0.0, 0.0, 2)

    # Sibling commits in the same fragment arbitrate symmetrically regardless of tuple order.
    for commit_pair in (
        (StateEffect("kv", "commit", 1, arbiter="z"), StateEffect("kv", "commit", 2, arbiter="a")),
        (StateEffect("kv", "commit", 2, arbiter="a"), StateEffect("kv", "commit", 1, arbiter="z")),
    ):
        sib = FragmentConductor(Engine([], backend="python"), state={"kv": 0})
        ev = sib.submit(ProgramFragment("sib", state_effects=(
            StateEffect("kv", "write", 1), StateEffect("kv", "write", 2), *commit_pair,
        ))).event
        sib.advance(until=(ev,))
        assert sib.state.committed_version("kv") == 1

    # Serial zero-duration write+commit chain past default version limit reclaims superseded versions,
    # and repeated same-version causal commits keep event-commit ancestry O(1).
    chain = FragmentConductor(Engine([], backend="python"), state={"kv": 0})
    prev = None
    for ver in range(1, 70):
        prev = chain.submit(ProgramFragment(f"c{ver}", state_effects=(
            StateEffect("kv", "write", ver), StateEffect("kv", "commit", ver),
        )), after=(prev,) if prev is not None else ()).event
    chain.advance(until=(prev,))
    assert chain.state.committed_version("kv") == 69
    assert len(chain.state.live_versions("kv")) <= 2

    same = FragmentConductor(Engine([], backend="python"), state={"kv": 0})
    same_prev = same.submit(ProgramFragment("init", state_effects=(
        StateEffect("kv", "write", 1), StateEffect("kv", "commit", 1),
    ))).event
    for idx in range(70):
        same_prev = same.submit(ProgramFragment(f"s{idx}", state_effects=(
            StateEffect("kv", "commit", 1),
        )), after=(same_prev,)).event
    same.advance(until=(same_prev,))
    assert same.state.committed_version("kv") == 1
    assert len(same.state._event_commits) <= 2
