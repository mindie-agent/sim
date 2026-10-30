"""Independent causal checks for the device invocation boundary."""
import pytest

from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.invocation import InvocationProgram, InvocationTiming
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage


class RecordingBody:
    def __init__(self, work):
        self.work = work
        self.calls = []
        self.events = []

    def start(self, engine, name, ready, *, owner=""):
        self.calls.append((engine.now, ready.done, owner))
        event = engine.submit(name + ":service", self.work, after=(ready,), owner=owner)
        self.events.append(event)
        return event


class WideBody:
    def __init__(self, phases):
        self.phases = phases
        self.calls = []

    def start(self, engine, name, ready, *, owner=""):
        self.calls.append(engine.now)
        events = tuple(
            engine.submit(f"{name}:phase{index}", {"cube": 1}, after=(ready,), owner=owner)
            for index in range(self.phases)
        )
        return engine.submit(name + ":complete", after=events, owner=owner)


class CountingProgram:
    def __init__(self, program):
        self.program = program
        self.calls = []

    def start(self, engine, name, ready, *, owner=""):
        self.calls.append(engine.now)
        return self.program.start(engine, name, ready, owner=owner)


class EmptyBody:
    def __init__(self):
        self.calls = 0

    def start(self, engine, name, ready, *, owner=""):
        self.calls += 1
        return engine.submit(name + ":empty", after=(ready,), owner=owner)


def test_timing_defaults_and_round_trip_preserve_provenance_without_inventing_cost():
    assert InvocationTiming() == InvocationTiming(0.0, 0.0, {})
    source = {"source": "capture", "grouped_residual_ms": 40}
    declared = InvocationTiming.from_mapping({"provenance": source})
    assert declared.startup_ms == declared.minimum_ms == 0
    assert declared.to_dict() == {
        "startup_ms": 0.0, "minimum_ms": 0.0, "provenance": source,
    }
    source["source"] = "mutated"
    assert declared.provenance["source"] == "capture"

    mapped = InvocationTiming.from_mapping({
        "startup_ms": 3, "minimum_ms": 5, "provenance": {"run": "r1"},
    })
    assert mapped.to_dict() == {
        "startup_ms": 3.0, "minimum_ms": 5.0, "provenance": {"run": "r1"},
    }


def test_startup_delays_body_after_original_readiness_and_forwards_owner():
    engine = Engine([Resource("cube", 10)], trace_limit=100)
    ready = engine.submit("input_ready", release=10)
    body = RecordingBody({"cube": 20})
    done = InvocationProgram(body, InvocationTiming(startup_ms=5)).start(
        engine, "inv", ready, owner="op")
    engine.advance(until=(ready,))
    assert engine.now == 10
    assert body.calls == []
    engine.advance(horizon=14.5)
    assert body.calls == []

    engine.advance(until=(done,))
    assert body.calls == [(15.0, True, "op")]
    assert body.events[0].time == 17
    service = next(row for row in engine.trace if row["task"] == "inv:body:service")
    assert (service["start_ms"], service["end_ms"]) == (15, 17)
    assert done.time == 17
    assert engine.serviced["cube"] == 20
    assert engine.busy_ms["cube"] == 2


@pytest.mark.parametrize(
    "startup,minimum,expected",
    ((5, 7, 9), (5, 20, 20), (0, 4, 4), (2, 0, 6)),
)
def test_minimum_is_concurrent_with_startup_plus_body(startup, minimum, expected):
    engine = Engine([Resource("cube", 10)])
    ready = engine.submit("ready")
    body = RecordingBody({"cube": 40})
    done = InvocationProgram(body, InvocationTiming(startup, minimum)).start(
        engine, "inv", ready)
    engine.advance(until=(done,))
    assert done.time == pytest.approx(expected)
    assert len(body.calls) == 1
    assert engine.serviced["cube"] == 40
    assert engine.busy_ms["cube"] == pytest.approx(4)
    assert engine.summary()["live_tasks"] == 0
    assert engine.summary()["live_programs"] == 0


def test_two_independent_invocations_do_not_gain_a_global_barrier():
    engine = Engine([Resource("left", 10), Resource("right", 10)])
    ready = engine.submit("ready")
    left, right = RecordingBody({"left": 10}), RecordingBody({"right": 10})
    first = InvocationProgram(left, InvocationTiming(startup_ms=5)).start(
        engine, "first", ready, owner="a")
    second = InvocationProgram(right, InvocationTiming(startup_ms=5)).start(
        engine, "second", ready, owner="b")
    engine.advance(until=(first, second), first=False)
    assert first.time == second.time == 6
    assert left.calls == [(5.0, True, "a")]
    assert right.calls == [(5.0, True, "b")]
    assert engine.serviced == {"left": 10, "right": 10}


def test_body_expansion_waits_for_startup_and_keeps_idle_state_bounded():
    engine = Engine([Resource("cube", 10)])
    ready = engine.submit("ready")
    body = WideBody(64)
    done = InvocationProgram(body, InvocationTiming(startup_ms=10)).start(
        engine, "wide", ready)
    engine.advance(horizon=5)
    assert body.calls == []
    assert engine.summary()["live_tasks"] == 1
    assert engine.summary()["peak_live_tasks"] <= 2

    engine.advance(until=(done,))
    assert body.calls == [10.0]
    assert engine.serviced["cube"] == 64
    assert engine.summary()["live_tasks"] == 0
    assert engine.summary()["live_programs"] == 0


def test_nested_phase_pipeline_is_charged_startup_once():
    def run(wrapped):
        engine = Engine([Resource("load", 10), Resource("cube", 10)], trace_limit=4096)
        ready = engine.submit("ready")
        body = CountingProgram(PipelineProgram(
            20, 2, lambda index: (
                Stage("load", {"load": 2}),
                Stage("compute", {"cube": 3}, after=("load",)),
            ),
        ))
        if wrapped:
            done = InvocationProgram(body, InvocationTiming(startup_ms=5)).start(
                engine, "pipe", ready)
        else:
            done = body.start(engine, "pipe", ready)
        engine.advance(until=(done,))
        return engine, done.time, body.calls

    baseline, baseline_ms, baseline_calls = run(False)
    wrapped, wrapped_ms, wrapped_calls = run(True)
    assert baseline_calls == [0.0]
    assert wrapped_calls == [5.0]
    assert wrapped_ms == pytest.approx(baseline_ms + 5)
    assert wrapped.serviced == baseline.serviced
    assert sum(row["task"].endswith(":startup") for row in wrapped.trace) == 1
    assert wrapped.summary()["live_tasks"] == 0


def test_wrapper_preserves_physical_work_allocation_and_completion_floor():
    def run(wrapped):
        engine = Engine(
            [Resource("hbm", 20), Resource("cube", 10)], pools={"L1": 64},
            pool_units={"L1": 1}, trace_limit=100)
        ready = engine.submit("ready")
        body = KernelProgram((
            Stage("load", {"hbm": 40}, acquire=(Allocation("buf", "L1", 64, (0,)),)),
            Stage("compute", {"cube": 40}, after=("load",),
                  free=(Allocation("buf", "L1", 64, (0,)),)),
        ))
        if wrapped:
            done = InvocationProgram(
                body, InvocationTiming(startup_ms=3, minimum_ms=6)).start(
                    engine, "inv", ready)
        else:
            done = body.start(engine, "inv", ready)
        engine.advance(until=(done,))
        return engine, done.time

    plain, plain_ms = run(False)
    wrapped, wrapped_ms = run(True)
    assert plain_ms == 6
    assert wrapped_ms == 9  # max(floor=6, startup+body=3+6), not 3+max(6,6).
    assert wrapped.serviced == plain.serviced == {"hbm": 40, "cube": 40}
    assert wrapped.busy_ms == plain.busy_ms == {"hbm": 2, "cube": 4}
    assert wrapped.peak_allocated == plain.peak_allocated == {"L1": 64}
    assert wrapped.allocated == {"L1": 0}
    assert wrapped.summary()["live_tasks"] == 0


def test_body_allocation_backpressure_remains_causal_under_startup():
    engine = Engine(
        [Resource("wire", 10), Resource("cube", 10)], pools={"L1": 64},
        pool_units={"L1": 1}, trace_limit=100)
    held = Allocation("held", "L1", 64, (0,))
    holder = engine.submit(
        "holder", {"wire": 100}, acquire=(held,), free=(held,))
    ready = engine.submit("ready")
    own = Allocation("body", "L1", 64, (0,))
    body = KernelProgram((
        Stage("service", {"cube": 20}, acquire=(own,), free=(own,)),
    ))
    done = InvocationProgram(body, InvocationTiming(startup_ms=2)).start(
        engine, "waiting", ready)
    engine.advance(until=(done,))
    assert holder.time == 10
    service = next(row for row in engine.trace if row["task"] == "waiting:body:service")
    assert (service["start_ms"], service["end_ms"]) == (10, 12)
    assert done.time == 12
    assert engine.serviced == {"wire": 100, "cube": 20}
    assert engine.allocated == {"L1": 0}
    assert engine.peak_allocated == {"L1": 64}


@pytest.mark.parametrize("startup", (0, 2))
def test_unresolved_readiness_keeps_the_body_unexpanded(startup):
    engine = Engine([Resource("cube", 10)])
    ready = engine.future("input")
    body = WideBody(8)
    done = InvocationProgram(body, InvocationTiming(startup_ms=startup)).start(
        engine, "waiting", ready)
    engine.advance(horizon=5)
    assert engine.now == 5
    assert body.calls == []
    engine.resolve(ready)
    engine.advance(until=(done,))
    assert body.calls == [5.0 + startup]
    assert done.time == pytest.approx(5.8 + startup)
    assert engine.serviced["cube"] == 8


def test_zero_timing_keeps_the_body_boundary_unchanged():
    engine = Engine([Resource("cube", 10)], trace_limit=100)
    ready = engine.submit("ready", release=7)
    body = RecordingBody({"cube": 20})
    done = InvocationProgram(body, InvocationTiming()).start(engine, "plain", ready)
    engine.advance(until=(done,))
    assert body.calls == [(7.0, True, "")]
    service = next(row for row in engine.trace if row["task"] == "plain:body:service")
    assert (service["start_ms"], service["end_ms"]) == (7, 9)
    assert done.time == 9
    assert engine.serviced["cube"] == 20


@pytest.mark.parametrize("startup,minimum,expected", ((0, 0, 0), (3, 5, 5)))
def test_empty_body_is_an_explicit_declared_invocation(startup, minimum, expected):
    engine = Engine([Resource("cube", 10)])
    ready = engine.submit("ready")
    body = EmptyBody()
    done = InvocationProgram(body, InvocationTiming(startup, minimum)).start(
        engine, "empty", ready)
    engine.advance(until=(done,))
    assert body.calls == 1
    assert done.time == expected
    assert engine.serviced["cube"] == 0
    assert engine.summary()["live_tasks"] == 0
    assert engine.summary()["live_programs"] == 0
