"""C7 context never creates readiness; actual pipeline events do."""
import pytest

from mindie_sim.prediction.collective_readiness import (
    bind_pipeline_ready, start_with_pipeline,
)
from mindie_sim.prediction.collectives import RingGroup
from mindie_sim.prediction.communication import MessageFlow, RoutedHop, RoutedMessage
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.schema.common import ModelingError


def _report():
    calls = []
    for rank in (0, 1):
        calls.append({
            "rank_id": f"rank_{rank}", "rank_index": rank,
            "operation": {"dtype": "BFP16", "typed_bytes": 10240},
            "entry": {"start_ns": 1_000_000_000 + rank},
            "completion": {"end_ns": 1_009_000_000 + rank},
            "observed_context": {"entry_domain": "rank_local",
                                 "cross_rank_aligned": False},
            "clock": {"domain": "rank_local", "cross_rank_aligned": False},
            "dependencies": (), "waits": (),
            "unknowns": ("cross_rank_clock_not_established",),
        })
    return {"schema": "mindie-profiler/collective-readiness/1", "calls": calls}


def _plan():
    return RingGroup((0, 1), 1, (
        RoutedMessage(MessageFlow(0, 1, 10240),
                      (RoutedHop("wire", 10240, 10240, None, None, 0.0),)),
        RoutedMessage(MessageFlow(1, 0, 10240),
                      (RoutedHop("wire", 10240, 10240, None, None, 0.0),)),
    ))


def _run(rank0_delay):
    engine = Engine([Resource("wire", 1_000_000_000, "bytes")], trace_limit=16)
    base = engine.submit("base")
    ready = {
        0: engine.submit("upstream0", after=(base,), latency=rank0_delay),
        1: engine.submit("upstream1", after=(base,), latency=0.0),
    }
    completions, context = start_with_pipeline(
        _report(), engine, _plan(), ready)
    engine.advance(until=tuple(completions.values()))
    return ({rank: event.time for rank, event in completions.items()},
            context, dict(engine.summary()["serviced"]))


def test_pipeline_predecessor_controls_completion_not_c7_chronology():
    base, _, base_service = _run(0.0)
    delayed, context, delayed_service = _run(5.0)
    assert base_service == delayed_service == {"wire": 20480}
    assert delayed[0] > base[0] + 4.9 and delayed[1] > base[1] + 4.9
    assert all(item["readiness_basis"] == "pipeline_engine_event"
               for item in context.values())
    assert context[0]["entry_ns"] == 1_000_000_000


def test_c7_requires_actual_ready_coverage_and_typed_call():
    engine = Engine([Resource("wire", 1_000_000_000, "bytes")], trace_limit=16)
    base = engine.submit("base")
    with pytest.raises(ModelingError, match="collective_readiness_ready_coverage"):
        bind_pipeline_ready(_report(), {0: base})
    bad = _report()
    bad["calls"][0]["operation"].pop("typed_bytes")
    with pytest.raises(ModelingError, match="collective_readiness_typed_admission"):
        bind_pipeline_ready(bad, {0: base, 1: base})
