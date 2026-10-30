"""Repeated immutable stages preserve finite execution and active memory."""
from dataclasses import replace
import weakref

from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage


def _execute(*, reuse, trace_limit=1000):
    engine = Engine(
        [Resource("read", 12, units=2), Resource("cube", 8, units=2)],
        pools={"l1": 16}, pool_units={"l1": 2}, trace_limit=trace_limit,
    )
    ready = engine.submit("ready", latency=0.25)
    background = engine.submit("background", {"read": 7, "cube": 3}, release=1.5)
    templates = {}

    def tile(index):
        slot, width = index % 2, 1 if index == 8 else 4
        key = slot, width
        if key not in templates:
            allocation = Allocation(f"slot{slot}", "l1", 8, (slot,))
            templates[key] = (
                Stage("load", {"read": width * 6}, caps={"read": 5},
                      acquire=(allocation,), placement={"read": (slot,)}),
                Stage("compute", {"cube": width * 3}, after=("load",),
                      previous=("publish",), placement={"cube": (slot,)}, coupled=True),
                Stage("publish", after=("compute",), free=(allocation,),
                      program=KernelProgram((Stage("output", {"read": width}),))),
            )
        return templates[key] if reuse else tuple(replace(s) for s in templates[key])

    plan = PipelineProgram(9, 2, tile)
    complete = plan.start(engine, "pipeline", ready)
    concurrent = plan.start(engine, "concurrent", ready)
    engine.advance(until=(complete, concurrent, background), first=False)
    return engine.summary(), list(engine.trace)


def test_prepared_reuse_preserves_full_trace_under_contention_and_buffer_lifetimes():
    reused, fresh = _execute(reuse=True), _execute(reuse=False)
    assert reused == fresh
    assert reused[0]["live_tasks"] == reused[0]["live_programs"] == 0
    assert reused[0]["live_allocated"] == {"l1": 0}


def test_disabled_trace_retains_identical_global_service_and_event_execution():
    traced, events = _execute(reuse=True)
    quiet, no_events = _execute(reuse=True, trace_limit=0)
    assert events and not no_events
    for key in ("retained_trace_events", "trace_dropped"):
        traced.pop(key)
        quiet.pop(key)
    assert traced == quiet


def test_prepared_allocation_totals_recheck_live_occupancy_for_each_tile():
    engine = Engine([Resource("read", 1)], pools={"l1": 10},
                    pool_units={"l1": 2}, trace_limit=100)
    held = Allocation("background", "l1", 4, (0,))
    engine.submit("background", acquire=(held,), free=(held,), latency=3)
    ready = engine.submit("ready")
    operands = (Allocation("a", "l1", 4, (0, 1)), Allocation("b", "l1", 3, (0,)))
    stages = (Stage("copy", {"read": 1}, acquire=operands, free=operands),)
    complete = PipelineProgram(3, 1, lambda _: stages).start(engine, "p", ready)
    engine.advance(until=(complete,))
    copies = [row for row in engine.trace if row["task"].endswith(":copy")]
    # Each operand fits beside the background separately, but their common
    # unit needs 7 bytes and must wait for the background's four bytes to free.
    assert [(row["start_ms"], row["end_ms"]) for row in copies] == [(3, 4), (4, 5), (5, 6)]
    assert complete.time == 6 and engine.serviced == {"read": 3}
    assert engine.summary()["peak_allocated_per_unit"] == {"l1:unit0": 7, "l1:unit1": 4}
    assert engine.summary()["live_allocated"] == {"l1": 0}


def test_prepared_submission_keeps_duplicate_pending_and_resolved_dependencies():
    engine = Engine([Resource("read", 10)], trace_limit=100)
    prior = engine.submit("prior", {"read": 10})
    engine.advance(until=(prior,))
    promised = engine.future("promised")
    gate = engine.submit("gate", latency=2, on_complete=lambda _: engine.resolve(promised))
    done = engine.submit("consumer", {"read": 20}, release=4,
                         after=(prior, promised, gate, promised, prior))
    engine.advance(until=(done,))
    assert prior.time == 1 and gate.time == promised.time == 3 and done.time == 6
    assert engine.serviced == {"read": 30}
    assert engine.summary()["completed_tasks"] == 3
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_stage_work_and_placement_are_construction_time_snapshots():
    work, caps, units = {"read": 12}, {"read": 3}, [0]
    stage = Stage("read", work, caps=caps, placement={"read": units})
    work["read"], caps["read"], units[0] = 120, 30, 1
    engine = Engine([Resource("read", 12, units=2)])
    ready = engine.submit("ready")
    complete = PipelineProgram(3, 1, lambda _: (stage,)).start(engine, "p", ready)
    engine.advance(until=(complete,))
    assert complete.time == 12
    assert engine.serviced == {"read": 36}


def test_generated_stages_leave_only_the_live_slot_window():
    refs, retained = [], []

    def tile(index):
        retained.append(sum(ref() is not None for ref in refs))
        stage = Stage("varying", {"read": index + 1})
        refs.append(weakref.ref(stage))
        return (stage,)

    engine = Engine([Resource("read", 1)])
    ready = engine.submit("ready")
    complete = PipelineProgram(100, 3, tile).start(engine, "p", ready)
    engine.advance(until=(complete,))
    assert complete.time == 5050
    assert max(retained) <= 3
    assert not any(ref() is not None for ref in refs)
