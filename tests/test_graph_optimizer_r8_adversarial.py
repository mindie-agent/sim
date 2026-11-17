"""NR-S2-FIX8: Round-8 architectural-closure witnesses.

[Role] Reproduce the six independent Round-8 expected invariants: implicit
  positional data operands across live stream events and a nonzero-event-cost
  canonical Qwen run, both reused-``lease_token`` release variants retaining the
  newer generation, a positive small cap with an uncapped peer completing
  across scales, and deterministic bytecode tick spacing on a 3,000-input join
  plus a 3,000-consumer fan-out.
[Invariants] Data operands stay a complete ordered tensor expression while
  control events stay in ``deps``; a stale copy capability can never free a
  newer token generation; a representable positive cap advances instead of
  stalling unrelated flows; every audited loop ticks the shared counter.
[Timeouts & Locks] Pure in-process simulation; wide discovery runs unbudgeted
  for structural bytecode counts, never host-load timing.
[Side-Effect Policy] No filesystem, environment or device mutation.
"""
from __future__ import annotations

from dataclasses import replace
import dis
import gc
import sys

import pytest

from mindie_sim import compose_case
from mindie_sim.prediction import stream_discovery
from mindie_sim.prediction.canonical_optimizer import _run_canonical
from mindie_sim.prediction.fusion_bridge import (
    _strip_events, canonical_plan, canonical_stream_move, insert_stream_events,
)
from mindie_sim.prediction.fusion_graph import OpNode, TensorValue, TypedGraph
from mindie_sim.prediction.fusion_pool import (
    PoolHolder, PrefetchPoolProgram, ReleasePoolProgram,
)
from mindie_sim.prediction.plan_model import PlanNode
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.schema.operators import OpSpec

from graph_opt_fixtures import canonical_qwen, hw

_CANONICAL_INPUTS = (
    "s0.l0.beta_contiguous", "s0.l0.delta_gates",
    "s0.l0.q_flat", "s0.l0.qkv_concatenate",
)
_WIDE = 3000
_CASE = None


@pytest.fixture(scope="module", autouse=True)
def _release_wide_fixtures():
    """Collect this module's wide graphs before later timing-sensitive files."""
    yield
    gc.collect()


def case():
    global _CASE
    if _CASE is None:
        _CASE = compose_case("Qwen3.8-27B", hardware="a3", tp=1, layers=1)
    return _CASE


def _add(op_id, deps=(), *, family="add", metadata=(), hbm=1_000_000.0):
    # A caller event/barrier marker is a genuinely payload-free control; only
    # payload-carrying families get data work in these optimizer fixtures.
    control = family in {"barrier", "event"}
    return OpSpec(
        op_id=op_id, family=family, phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=(1, 4), local_shape=(1, 4),
        padded_shape=(1, 4), launch_shape=(1, 4),
        hbm_bytes=0.0 if control else hbm,
        vector_ops=0.0 if control else 2.0, dtype="bf16",
        deps=tuple(deps), metadata=tuple(metadata))


# ---------------------------------------------------------------------------
# Finding 1: implicit positional operands must precede the control rewrite
# ---------------------------------------------------------------------------
def test_r8_f1_raw_implicit_operands_survive_live_stream_events():
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _add("c", ("a", "analysis_input", "a")),
    ))
    expected = ("a:out", "analysis_input:source", "a:out")
    assert plan.graph.ops["c"].inputs == expected
    hardware = hw(max_streams=2, stream_switch_ms=0.01)
    moved = canonical_stream_move(plan, ("c",), hardware)
    generated = [op for op in moved if op.family == "event"]
    assert generated
    consumer = next(op for op in moved if op.op_id == "c")
    # A wait event is a control dependency; scheduling deps are deduplicated.
    assert consumer.deps == ("event:wait:a->c", "analysis_input")
    assert consumer.deps.count("event:wait:a->c") == 1
    inserted = plan.rewrite(moved)
    assert inserted.graph.ops["c"].inputs == expected
    assert inserted.graph.validate()
    # Reinsertion strips and regenerates the crossing without losing an operand.
    twice = insert_stream_events(moved, hardware)
    assert plan.rewrite(twice).graph.ops["c"].inputs == expected
    # Stripping the generated events restores the event-free projection.
    stripped = _strip_events(moved)
    assert plan.rewrite(stripped).graph.ops["c"].inputs == expected


def test_r8_f1_canonical_qwen_nonzero_event_costs_preserve_inputs():
    raw, ops = canonical_qwen()
    plan = canonical_plan(raw, ops)
    plan = replace(plan, hardware=replace(
        plan.hardware, event_record_ms=1e-6, event_wait_ms=1e-6))
    before = {op.op_id: plan.graph.ops[op.op_id].inputs for op in plan.ops}
    report = _run_canonical(plan, evaluation_budget=64, per_pass=32,
                            max_streams=None, wall_budget_s=30.0)
    graph = report.canonical_plan.graph
    graph.validate()
    assert any(op.family == "event" for op in report.canonical_plan.ops)
    for op_id in _CANONICAL_INPUTS:
        assert graph.ops[op_id].inputs == before[op_id]
    fused_ids = {"fused:" + "+".join(row["members"])
                 for row in report.applied
                 if row["pass"] in {"vector_fusion", "cube_epilogue"}}
    for op_id, original in before.items():
        if op_id not in graph.ops:
            continue  # contracted member
        node = graph.ops[op_id]
        if len(node.inputs) == len(original) \
                and all(old == new for old, new in zip(original, node.inputs)):
            continue
        # Only a contracted producer may rename an operand; the position,
        # arity and every unrelated input survive the event rewrite.
        assert len(node.inputs) == len(original), op_id
        for old, new in zip(original, node.inputs):
            if old == new:
                continue
            assert graph.values[new].producer in fused_ids, (op_id, old, new)
    assert report.final.duration_ms == pytest.approx(
        0.3065054662131517, rel=1e-9)


# ---------------------------------------------------------------------------
# Finding 2: immutable copy capability outranks a mutable token alias
# ---------------------------------------------------------------------------
def _lease_engine():
    return Engine([Resource("hbm", 1024, "bytes"), Resource("host", 1, "ms")],
                  pools={"l1": 2048}, pool_units={"l1": 1})


def _live_l1(trace, at):
    live = 0
    for row in trace:
        if float(row["time"]) > at + 1e-12:
            continue
        delta = float(row["size"]) * len(tuple(row["units"]))
        live += int(delta) if row["action"] == "acquire" else -int(delta)
    return live


def _reused_token_trace(explicit_handle):
    engine = _lease_engine()
    holder = PoolHolder("shared")
    copy = PrefetchPoolProgram(holder, 1024, 1, lease_token="raw-token")
    release = ReleasePoolProgram(holder, lease_token="raw-token")
    entry = engine.submit("entry")
    first = copy.start(engine, "trial", entry)
    first_free = release.start(engine, "trial:release", first)
    engine.advance(until=(first_free,), first=False)
    assert engine.summary()["live_allocated"] == {"l1": 0.0}
    second = copy.start(engine, "trial", entry)
    consumer = engine.submit("consumer", {"host": 10}, after=(second,))
    stale = release.start(engine, "trial:release", first,
                          copy_handle=first if explicit_handle else None)
    genuine = release.start(engine, "trial:release", consumer,
                            copy_handle=second)
    assert stale.name == "trial:release:noop"
    assert genuine.name == "trial:release:free"
    engine.advance(horizon=11.9, first=False)
    trace = list(engine.allocation_trace)
    assert _live_l1(trace, 2.0) == 1024
    assert _live_l1(trace, 11.9) == 1024
    engine.advance(until=(genuine,), first=False)
    trace = list(engine.allocation_trace)
    frees = [row["time"] for row in trace
             if row["action"] == "release" and row["name"] == "trial:pool"]
    assert frees == [1.0, 12.0]
    assert engine.summary()["live_allocated"] == {"l1": 0.0}
    # The reusable token label never retargets the newer live capability.
    grants = holder._bucket(engine)
    alias = grants.capabilities[("token", "raw-token")]
    assert alias == "grant:1"
    assert alias != grants.capabilities[("handle", second.sequence)]
    return frees


def test_r8_f2_reused_token_stale_ready_cannot_free_newer_generation():
    assert _reused_token_trace(False) == [1.0, 12.0]


def test_r8_f2_reused_token_stale_copy_handle_cannot_free_newer_generation():
    assert _reused_token_trace(True) == [1.0, 12.0]


# ---------------------------------------------------------------------------
# Additional R8 finding: positive cap progress without stalling peers
# ---------------------------------------------------------------------------
def test_r8_progress_small_cap_and_uncapped_peer_across_scales():
    for exponent in range(9, 16):
        scale = float(10 ** exponent)
        engine = Engine([Resource("hbm", scale, "bytes"),
                         Resource("vector", scale, "ops")])
        a = engine.submit("a", {"hbm": 1e6, "vector": 1e6},
                          caps={"hbm": 0.1}, coupled=True)
        b = engine.submit("b", {"hbm": 1e6})
        engine.advance(until=(a, b), first=False)
        assert a.time is not None and b.time is not None, scale
        assert b.time < 1e-2, (scale, b.time)
        assert a.time == pytest.approx(1e7, rel=1e-9), (scale, a.time)
        assert engine.summary()["live_tasks"] == 0
        assert engine.serviced["hbm"] == pytest.approx(2e6), scale
        assert engine.serviced["vector"] == pytest.approx(1e6), scale


# ---------------------------------------------------------------------------
# Finding 5: deterministic bytecode tick spacing on wide join/fan-out
# ---------------------------------------------------------------------------
def _or_opargs():
    found = set()
    for arg, (name, _symbol) in enumerate(getattr(dis, "_nb_ops", ())):
        if name in {"|", "|="}:
            found.add(arg)
    return found or {7, 20}


_BINARY_OP = dis.opmap.get("BINARY_OP")
_FOR_ITER = dis.opmap["FOR_ITER"]
_OR_OPARGS = _or_opargs()


def _tick_spacing(run, counter):
    state = {"count": 0, "maximum": 0, "ticks": 0}
    original = stream_discovery._Ticker.tick
    module_file = stream_discovery.__file__

    def tick(self):
        if state["ticks"]:
            state["maximum"] = max(state["maximum"], state["count"])
        state["count"] = 0
        state["ticks"] += 1
        return original(self)

    def tracer(frame, event, arg):
        if event == "call":
            if frame.f_code.co_filename == module_file \
                    and frame.f_code.co_name != "tick":
                frame.f_trace_opcodes = True
                return tracer
            return None
        if event == "opcode" and frame.f_code.co_filename == module_file:
            counter(frame, state)
        return tracer

    stream_discovery._Ticker.tick = tick
    previous_trace = sys.gettrace()
    sys.settrace(tracer)
    try:
        result = run()
    finally:
        sys.settrace(previous_trace)
        stream_discovery._Ticker.tick = original
    # The trailing interval has no following tick; only consecutive pairs count.
    return result, state["maximum"], state["ticks"]


def _count_or(frame, state):
    code = frame.f_code.co_code
    if code[frame.f_lasti] == _BINARY_OP \
            and code[frame.f_lasti + 1] in _OR_OPARGS:
        state["count"] += 1


def _count_for_iter(frame, state):
    if frame.f_code.co_code[frame.f_lasti] == _FOR_ITER:
        state["count"] += 1


def _wide_fan_in():
    return tuple(
        PlanNode(f"n{index}", (("vector", 1.0),)) for index in range(_WIDE)
    ) + (PlanNode("join", (("vector", 1.0),),
                  deps=tuple(f"n{index}" for index in range(_WIDE))),)


def _wide_fan_out_graph():
    values = {
        "v": TensorValue("v", "bf16", (1, 4), producer="root",
                         storage_bytes=8),
        "out": TensorValue("out", "bf16", (1, 4), producer="n0",
                           storage_bytes=8),
    }
    ops = {"root": OpNode(op_id="root", kind="add", resource="vector",
                          inputs=(), outputs=("v",),
                          work=(("vector", 1.0),))}
    for index in range(_WIDE):
        ops[f"n{index}"] = OpNode(
            op_id=f"n{index}", kind="add", resource="vector", inputs=("v",),
            outputs=(("out",) if index == 0 else ()),
            work=(("vector", 1.0),))
    return TypedGraph(1, values, ops, ("out",))


def test_r8_f5_tick_spacing_on_wide_join_and_fan_out():
    plan = _wide_fan_in()
    # No wall/RSS budget: spacing is a structural bytecode property and must
    # not depend on host load or a stopwatch race.
    regions, join_max, join_ticks = _tick_spacing(
        lambda: stream_discovery.stream_regions(plan, 2, None), _count_or)
    assert isinstance(regions, tuple) and len(regions) == _WIDE
    assert join_ticks > _WIDE
    assert join_max <= 4, join_max

    graph = _wide_fan_out_graph()
    result, fan_max, fan_ticks = _tick_spacing(
        lambda: stream_discovery.index_graph(graph, None), _count_for_iter)
    successors, consumers = result
    assert successors is not None and consumers is not None
    assert len(consumers["v"]) == _WIDE
    assert fan_ticks > _WIDE
    assert fan_max <= 4, fan_max
