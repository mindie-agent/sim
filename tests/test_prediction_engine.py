"""Small independent timing oracles for the replacement execution engine."""
import pytest

from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.program import PipelineProgram, Stage


def test_dependency_stream_and_independent_resource_overlap():
    engine = Engine([Resource("cube", 10), Resource("hbm", 20)])
    load = engine.submit("load", {"hbm": 100})
    independent = engine.submit("other", {"cube": 40})
    compute = engine.submit("compute", {"cube": 60}, after=(load,))
    engine.advance(until=(compute, independent), first=False)
    assert load.time == 5
    assert independent.time == 4
    assert compute.time == 11
    assert engine.serviced == {"cube": 100, "hbm": 100}
    assert engine.summary()["live_tasks"] == 0


def test_fair_service_rebalances_when_a_short_transfer_finishes():
    engine = Engine([Resource("wire", 100)])
    short = engine.submit("short", {"wire": 100})
    long = engine.submit("long", {"wire": 300})
    engine.advance(until=(short, long), first=False)
    assert short.time == 2
    assert long.time == 4
    assert engine.serviced["wire"] == 400


def test_limited_outstanding_supply_leaves_capacity_for_other_streams():
    engine = Engine([Resource("hbm", 100)])
    narrow = engine.submit("narrow", {"hbm": 100}, caps={"hbm": 20})
    wide = engine.submit("wide", {"hbm": 100})
    engine.advance(until=(narrow, wide), first=False)
    assert wide.time == 1.25
    assert narrow.time == 5


def test_host_submission_pipelines_with_device_completion():
    engine = Engine([Resource("host", 1, policy="fifo"), Resource("cube", 1)])
    device, submit = None, None
    for index in range(3):
        submit = engine.submit(f"submit{index}", {"host": 2}, after=(() if submit is None else (submit,)))
        device = engine.submit(f"kernel{index}", {"cube": 20},
                               after=(submit,) + (() if device is None else (device,)))
    engine.advance(until=(device,))
    assert device.time == 62


def test_allocation_lifetime_applies_backpressure_until_the_consumer_finishes():
    engine = Engine([Resource("load", 1), Resource("compute", 1)], pools={"L1": 128})
    completions = []
    for index in range(2):
        buffer = Allocation(f"tile{index}", "L1", 128)
        load = engine.submit(f"load{index}", {"load": 2}, acquire=(buffer,))
        compute = engine.submit(f"compute{index}", {"compute": 3}, after=(load,), free=(buffer,))
        completions.append(compute)
    engine.advance(until=completions, first=False)
    assert [event.time for event in completions] == [5, 10]
    assert engine.peak_allocated == {"L1": 128}
    assert engine.allocated == {"L1": 0}


def test_incremental_arrivals_do_not_restart_work_or_retain_completed_history():
    engine = Engine([Resource("wire", 10)], trace_limit=2)
    first = engine.submit("first", {"wire": 100})
    engine.advance(horizon=3)
    second = engine.submit("second", {"wire": 10})
    engine.advance(until=(second,))
    assert second.time == 5
    engine.advance(until=(first,))
    assert first.time == 11
    for index in range(1000):
        event = engine.submit(str(index), {"wire": 10}, after=(first,))
        engine.advance(until=(event,))
        first = event
    assert engine.summary()["live_tasks"] == 0
    assert engine.summary()["peak_live_tasks"] == 2
    assert len(engine.trace) == 2
    assert len(first._waiters) == 0


def test_observation_envelope_does_not_fabricate_resource_work():
    engine = Engine([Resource("cube", 10)])
    event = engine.submit("kernel", {"cube": 10}, latency=3)
    consumer = engine.submit("consumer", {"cube": 10}, after=(event,))
    engine.advance(until=(consumer,))
    assert event.time == 3
    assert consumer.time == 4
    assert engine.serviced["cube"] == 20


def test_tiny_positive_work_and_absolute_arrival_offset():
    for offset in (0, 1000):
        engine = Engine([Resource("wire", 1)])
        engine.now = offset
        event = engine.submit("tiny", {"wire": 2e-9})
        engine.advance(until=(event,))
        assert event.time - offset == pytest.approx(2e-9, rel=1e-4)
        assert engine.serviced["wire"] == 2e-9


def test_lazy_finite_pipeline_includes_fill_drain_and_bounded_buffer_slots():
    for slots, expected in ((1, 50_000), (2, 30_002)):
        engine = Engine([Resource("load", 1), Resource("cube", 1)])
        ready = engine.submit("ready")
        plan = PipelineProgram(10_000, slots, lambda i: (
            Stage("load", {"load": 2}), Stage("compute", {"cube": 3}, after=("load",)),
        ))
        complete = plan.start(engine, "gemm", ready)
        engine.advance(until=(complete,))
        assert complete.time == expected
        assert engine.serviced == {"load": 20_000, "cube": 30_000}
        assert engine.summary()["peak_live_tasks"] <= 8
        assert engine.summary()["live_programs"] == 0
