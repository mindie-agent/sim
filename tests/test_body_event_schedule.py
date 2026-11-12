"""Exact event-scheduler order for the native body graph.

Zero-work chains must keep sequence order and the same completion instants
without rescanning unrelated active work at every hop.
"""
import math

from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.body import body_forward
from test_prediction_system import case


def _engine():
    return Engine(
        [Resource("compute", 10.0, units=2), Resource("queue", 1.0, policy="fifo")],
        pools={"hbm": 100.0}, pool_units={"hbm": 1},
    )


def test_zero_work_chain_matches_serial_latency_and_stays_fast():
    engine = _engine()
    previous = engine.submit("origin", latency=1.5)
    for index in range(4000):
        previous = engine.submit(f"hop{index}", after=(previous,))
    tail = engine.submit("tail", {"compute": 20.0}, after=(previous,))
    engine.advance(until=(tail,))
    assert previous.time == 1.5
    assert tail.time == 1.5 + 2.0
    assert engine.summary()["live_tasks"] == 0
    assert engine.serviced["compute"] == 20.0


def test_same_timestamp_completion_order_is_sequence_order():
    engine = _engine()
    seen = []
    first = engine.submit("first", latency=1.0, on_complete=lambda event: seen.append(event.name))
    second = engine.submit("second", latency=1.0, on_complete=lambda event: seen.append(event.name))
    engine.advance(until=(first, second), first=False)
    assert seen == ["first", "second"]
    assert first.time == second.time == 1.0


def test_fifo_and_pool_admission_keep_lower_sequence_priority():
    engine = _engine()
    earlier = engine.submit("earlier", {"queue": 4.0})
    later = engine.submit("later", {"queue": 1.0})
    engine.advance(until=(later,), first=False)
    assert earlier.time == 4.0
    assert later.time == 5.0

    pooled = _engine()
    holder = pooled.submit("holder", acquire=(Allocation("live", "hbm", 80.0),), latency=2.0,
                           free=(Allocation("live", "hbm", 80.0),))
    waiting = pooled.submit("waiting", acquire=(Allocation("next", "hbm", 80.0),), latency=1.0,
                            free=(Allocation("next", "hbm", 80.0),))
    pooled.advance(until=(holder, waiting), first=False)
    assert holder.time == 2.0
    assert waiting.time == 3.0
    assert math.isclose(pooled.peak_allocated["hbm"], 80.0)


def test_fair_staggered_work_matches_closed_form():
    engine = Engine([Resource("compute", 100.0)])
    amounts = [1.0 + index * 0.01 for index in range(32)]
    events = [engine.submit(f"t{index}", {"compute": amount}) for index, amount in enumerate(amounts)]
    engine.advance(until=tuple(events), first=False)
    remaining = list(amounts)
    now = 0.0
    while any(remaining):
        alive = [value for value in remaining if value > 0]
        rate = 100.0 / len(alive)
        step = min(value / rate for value in alive)
        now += step
        remaining = [value - step * rate if value > 0 else 0.0 for value in remaining]
    assert events[-1].time == pytest_approx(now)
    assert engine.serviced["compute"] == pytest_approx(sum(amounts))


def pytest_approx(value):
    from pytest import approx
    return approx(value)


def test_body_forward_detail_preserves_timeline_and_is_deterministic():
    raw = case(output=1)
    options = dict(phase="decode", tokens=1, context=32, sequences=1)
    first = body_forward(raw, **options)
    second = body_forward(raw, **options)
    assert first["body_forward_ms"] == second["body_forward_ms"]
    assert first["timeline"] == second["timeline"]
    assert first["operator_occurrences"] == sum(row["family"] != "barrier" for row in first["timeline"])
    changed = body_forward(raw, phase="decode", tokens=2, context=32, sequences=1)
    assert changed["geometry"]["tokens_per_sequence"] == 2
    assert changed["body_forward_ms"] != first["body_forward_ms"]
