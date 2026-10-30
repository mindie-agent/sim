"""Finite transfer tests distinguish credit stalls from steady supply limits."""
import pytest

from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.transfer import FiniteTransfer
from tools.finite_transfer_oracle import finite_transfer


def simulate(amount, slots, response=4, rate=10, granule=10, *, trace=0):
    engine = Engine((Resource("wire", rate, "bytes"),), trace_limit=trace)
    ready = engine.submit("ready")
    done = FiniteTransfer({"wire": amount}, granule, slots, response).start(
        engine, "copy", ready)
    engine.advance(until=(done,))
    return done.time, engine


def test_credit_reuse_depends_on_response_and_tail_includes_final_return():
    # Wire service is 1ms. Two slots yield finishes at 5,6,10,11,15ms.
    duration, engine = simulate(50, 2, trace=100)
    assert duration == 15
    returns = [row["end_ms"] for row in engine.trace
               if row["task"].endswith(":response")]
    assert returns == [5, 6, 10, 11, 15]
    assert engine.serviced["wire"] == 50
    assert simulate(50, 1)[0] == 25
    assert simulate(50, 8)[0] == simulate(50, None)[0] == 9
    assert simulate(10, 1)[0] == simulate(10, 8)[0] == 5


def test_real_tail_is_transaction_rounded_and_unlimited_service_is_exact():
    for slots in (1, 2, None):
        duration, engine = simulate(41, slots, response=0)
        assert duration == 5
        assert engine.serviced["wire"] == 50
    assert simulate(41, 2)[0] == simulate(50, 2)[0] == 15
    assert simulate(40, 2)[0] == 11


def test_other_transfers_use_capacity_while_credit_waits():
    engine = Engine((Resource("wire", 10, "bytes"),), trace_limit=100)
    ready = engine.submit("ready")
    slow = FiniteTransfer({"wire": 20}, 10, 1, 4).start(engine, "slow", ready)
    other = engine.submit("other", {"wire": 30}, release=1)
    engine.advance(until=(slow, other), first=False)
    assert other.time == 4
    assert slow.time == 10
    assert engine.serviced["wire"] == 50
    service = [row for row in engine.trace
               if row["task"].endswith(":service") and row["demand"]]
    assert [(row["start_ms"], row["end_ms"]) for row in service] == [(0, 1), (5, 6)]


def test_concurrent_invocations_share_real_service_and_keep_separate_credits():
    engine = Engine((Resource("wire", 10, "bytes"),))
    ready = engine.submit("ready")
    ends = [FiniteTransfer({"wire": 20}, 10, 1, 4).start(engine, str(i), ready)
            for i in range(2)]
    engine.advance(until=ends, first=False)
    assert [end.time for end in ends] == [12, 12]
    assert engine.serviced["wire"] == 40


def test_multiple_paths_have_independent_slots_and_join_after_the_slowest():
    engine = Engine((Resource("a", 10, "bytes"), Resource("b", 5, "bytes")))
    ready = engine.submit("ready")
    done = FiniteTransfer({"a": 50, "b": 25}, 10, 2, 4).start(engine, "copy", ready)
    engine.advance(until=(done,))
    assert done.time == 15  # a:15ms, b:12ms; no invented hop dependency.
    assert engine.serviced == {"a": 50, "b": 30}


def test_live_queue_is_bounded_independently_of_transfer_length():
    _, short = simulate(100, 3)
    _, long = simulate(100_000, 3)
    assert long.peak_tasks == short.peak_tasks
    assert long.peak_tasks <= 2 * 3 + 2
    assert long.summary()["live_tasks"] == long.summary()["live_programs"] == 0


def test_engine_matches_independent_serial_server_credit_recurrence():
    # Vary tails, saturation regimes and slots beyond the number of requests.
    for payload in (1, 10, 41, 100, 231):
        for slots in (1, 2, 3, 32):
            for rate in (2, 10, 100):
                for response in (0, .3, 4):
                    expected = finite_transfer(payload, 10, rate, slots, response)
                    actual, engine = simulate(payload, slots, response, rate)
                    assert actual == pytest.approx(expected["duration_ms"])
                    assert engine.serviced["wire"] == expected["charged_bytes"]


def test_finite_fifo_queue_preserves_transaction_interleaving_without_response_delay():
    engine = Engine((Resource("wire", 10, "bytes", policy="fifo"),))
    ready = engine.submit("ready")
    done = FiniteTransfer({"wire": 50}, 10, 1, 0).start(engine, "copy", ready)
    engine.advance(horizon=.5)
    other = engine.submit("other", {"wire": 10})
    engine.advance(until=(done, other), first=False)
    assert other.time == 2  # FIFO transaction0, other, transaction1..4.
    assert done.time == 6
