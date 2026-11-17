"""NR-S2-FIX10: Round-10 contraction-boundary and cross-tier conservation.

[Role] Reproduce the two Round-10 architectural findings: a contraction loses
  declared control classification on both boundary directions for every
  supported control metadata key, and a per-tier residual carries floating
  point drift into lower priority tiers.
[Invariants] A member's external control predecessor stays classified on the
  fused operator; a genuine external data producer is never deleted even when
  another member declares it as a control; every surviving consumer's ``deps``
  and all five control keys are rewritten to the fused id with first-occurrence
  deduplication; event insertion/reinsertion/stripping round trips preserve the
  same operand and control semantics.  Across priority tiers the original
  physical unit capacities bound one cumulative ``math.fsum`` ledger, so the
  public ``service_of`` sum never exceeds capacity by even one ULP.
  R16 explicit tensor producers complete before consumers become ready, even
  without scheduling deps; execution lowering never changes caller operands.
  R17 external sources and resolved state writers also gate readiness; state
  edges without scheduled order cannot cross streams or stages.
[Timeouts & Locks] Pure in-process simulation; no clocks, sockets or locks.
[Side-Effect Policy] No filesystem, environment or device mutation.
"""
from __future__ import annotations

import math
from collections import UserDict
from dataclasses import replace
from types import MappingProxyType

import pytest

from mindie_sim.prediction.fusion_bridge import (
    _strip_events, add_prefetch, apply_prefetch_rewrite, apply_rewrite,
    canonical_plan, canonical_stream_move, insert_stream_events, rewrite_fusion,
    settle_prefetch_rewrite,
)
from mindie_sim.prediction.fusion_graph import FeasibilityError
from mindie_sim.prediction.fusion_ports import full_operand_ports, input_port_entries
from mindie_sim.prediction.fusion_recipe import (
    RejectedFusion, synthesize_vector,
)
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.weight_prefetch import PrefetchDecision
from mindie_sim.schema.operators import OpSpec

from graph_opt_fixtures import hw
from test_graph_optimizer_r8_adversarial import _add, case

_KEYS = ("control_deps", "control_inputs", "barrier_after", "event_deps",
         "barrier_deps")
_CAPACITIES = (125548680209.0561, 4121231867897.778)
_TIERS = (100, 300, 1_000, 10_000)


def _declare(key, *names):
    return ((key, tuple(names)),)


def _contract(plan, members):
    candidate = synthesize_vector(
        plan.graph, members, tile_rows=1,
        capacity_per_owner=plan.hardware.ub_bytes_per_aiv)
    assert not isinstance(candidate, RejectedFusion), candidate.reasons
    return apply_rewrite(plan, rewrite_fusion(plan, candidate))


def _incoming_plan(key):
    return canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _add("b", ("analysis_input",)),
        _add("c", ("b", "a"), metadata=_declare(key, "a")),
        _add("d", ("c",)),
    ))


def _outgoing_plan(key, *, dual=False):
    data = ("analysis_input", "b") if dual else ("analysis_input", "a")
    return canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _add("b", ("a",), family="rotary",
             metadata=(("rotate_style", "interleaved"),)),
        _add("c", data, metadata=_declare(key, "a")),
    ))


def _assert_event_free_round_trip(plan, move_members, region, op_id,
                                  inputs, deps):
    hardware = hw(max_streams=2, stream_switch_ms=0.01)
    moved = canonical_stream_move(plan, move_members, hardware)
    twice = insert_stream_events(moved, hardware)
    stripped = _strip_events(twice)
    after = _contract(plan.rewrite(stripped), region)
    after.graph.validate()
    node = after.graph.ops[op_id]
    assert node.inputs == inputs
    assert node.deps == deps
    assert after.evaluate()["duration_ms"] > 0
    # Contract the moved graph first, then strip the generated events: the
    # restored scheduling edges must name the surviving fused producer.
    contracted = _contract(plan.rewrite(moved), region)
    cleaned = plan.rewrite(_strip_events(contracted.ops))
    cleaned.graph.validate()
    clean_node = cleaned.graph.ops[op_id]
    assert clean_node.inputs == inputs
    assert clean_node.deps == deps
    assert cleaned.evaluate()["duration_ms"] > 0


# ---------------------------------------------------------------------------
# Finding A: incoming control classification is carried onto the fused node
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", _KEYS)
def test_r10_fa_incoming_control_stays_control_on_fused_node(key):
    plan = _incoming_plan(key)
    assert plan.graph.ops["c"].inputs == ("b:out",)
    after = _contract(plan, ("c", "d"))
    after.graph.validate()
    fused = after.graph.ops["fused:c+d"]
    assert fused.inputs == ("b:out",)
    assert fused.deps == ("b", "a")
    metadata = dict(next(op for op in after.ops if op.op_id == "fused:c+d").metadata)
    assert metadata["control_deps"] == ("a",)
    assert after.evaluate()["duration_ms"] > 0
    _assert_event_free_round_trip(
        plan, ("c", "d"), ("c", "d"), "fused:c+d", ("b:out",), ("b", "a"))


@pytest.mark.parametrize("key", _KEYS)
def test_r10_fa_incoming_data_and_control_producer_is_never_deleted(key):
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _add("c", ("a",)),
        _add("d", ("c",), metadata=_declare(key, "a")),
    ))
    after = _contract(plan, ("c", "d"))
    after.graph.validate()
    fused = after.graph.ops["fused:c+d"]
    assert fused.inputs == ("a:out",)
    assert fused.deps == ("a",)
    metadata = dict(next(op for op in after.ops if op.op_id == "fused:c+d").metadata)
    assert "a" not in dict(metadata).get("control_deps", ())
    assert after.evaluate()["duration_ms"] > 0


# ---------------------------------------------------------------------------
# Finding A: outgoing control-only consumers are rewritten without data ports
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("key", _KEYS)
def test_r10_fa_outgoing_control_only_consumer_is_rewritten(key):
    plan = _outgoing_plan(key)
    assert plan.evaluate()["duration_ms"] > 0
    after = _contract(plan, ("a", "b"))
    after.graph.validate()
    node = after.graph.ops["c"]
    assert node.deps == ("analysis_input", "fused:a+b")
    assert after.evaluate()["duration_ms"] > 0
    _assert_event_free_round_trip(
        plan, ("c",), ("a", "b"), "c", ("analysis_input:source",),
        ("analysis_input", "fused:a+b"))


@pytest.mark.parametrize("key", _KEYS)
def test_r10_fa_outgoing_data_and_control_keeps_genuine_data_port(key):
    plan = _outgoing_plan(key, dual=True)
    after = _contract(plan, ("a", "b"))
    after.graph.validate()
    node = after.graph.ops["c"]
    assert node.inputs == ("analysis_input:source", "fused:a+b:out")
    assert node.deps == ("analysis_input", "fused:a+b")
    assert after.evaluate()["duration_ms"] > 0
    _assert_event_free_round_trip(
        plan, ("c",), ("a", "b"), "c",
        ("analysis_input:source", "fused:a+b:out"),
        ("analysis_input", "fused:a+b"))


# ---------------------------------------------------------------------------
# Finding B: one cumulative fsum ledger across every priority tier
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("capacity", _CAPACITIES)
@pytest.mark.parametrize("count", _TIERS)
def test_r10_fb_public_tiers_never_exceed_physical_capacity(count, capacity):
    engine = Engine([Resource("hbm", capacity, "bytes"),
                     Resource("vector", capacity, "ops")])
    cap = capacity / (count + 1)
    events = tuple(engine.submit(
        str(index), {"hbm": capacity, "vector": capacity},
        coupled=True, priority=index,
        caps={"hbm": cap} if index < count - 1 else None)
        for index in range(count))
    engine.advance(horizon=1.0, first=False)
    assert engine.now == 1.0
    ulp = math.ulp(capacity)
    for name in ("hbm", "vector"):
        used = math.fsum(engine.service_of(event)[name] for event in events)
        assert math.isfinite(used)
        assert used <= capacity
        assert max(0.0, (used - capacity) / ulp) == 0.0
    engine.advance(until=events, first=False, max_steps=10)
    assert all(event.done for event in events)
    assert engine.summary()["live_tasks"] == 0


_R16_HARDWARE = hw(max_streams=2, stream_switch_ms=0.01)
_R16_PORTS = (("p", "p:out"), ("p", "p:out"))


class _R16PriorityOp(OpSpec):
    # OpSpec has no priority field: the executable Stage owns admission policy.
    @property
    def priority(self):
        return int(dict(self.metadata).get("test_priority", 0))


def _r16_phase(ops, phase):
    events = insert_stream_events(ops, _R16_HARDWARE)
    assert events != ops and insert_stream_events(events, _R16_HARDWARE) == events
    return {"raw": ops, "events": events, "stripped": _strip_events(events)}[phase]


def _r16_ordered(plan):
    """Check every projected internal tensor edge against actual readiness."""
    assert plan.graph.validate()
    result = plan.evaluate()
    rows = {row["op_id"]: row for row in result["timeline"]}
    for op in plan.graph.ops.values():
        for value_id in op.inputs:
            producer = plan.graph.values[value_id].producer
            if producer is not None:
                assert rows[op.op_id]["ready_ms"] >= rows[producer]["complete_ms"]
    return result, rows


@pytest.mark.parametrize("deps", ((), ("g",), ("p",)))
@pytest.mark.parametrize("phase", ("raw", "events", "stripped"))
def test_r16_nonroot_explicit_data_has_exact_causal_timeline(deps, phase):
    c = _add("c", deps, metadata=(("control_deps", ("g", "p")),
                                 ("fusion_input_ports", _R16_PORTS)))
    ops = (_add("g", ("analysis_input",)), _add("p", ("g",)), c,
           replace(_add("tail", ("c",)), stream="aux"))
    plan = canonical_plan(case(), ops)
    current = plan.rewrite(_r16_phase(ops, phase))
    result, rows = _r16_ordered(current)
    assert current.graph.ops["c"].inputs == ("p:out", "p:out")
    assert current.graph.ops["c"].semantic_complete
    assert (rows["p"]["ready_ms"], rows["p"]["complete_ms"]) == (0.0025, 0.005)
    assert (rows["c"]["ready_ms"], rows["c"]["complete_ms"]) == (0.005, 0.0075)
    assert result["duration_ms"] == (0.02 if phase == "events" else 0.01)
    assert next(op for op in current.ops if op.op_id == "c") is c
    assert plan.ops is ops and c.deps == deps and input_port_entries(c.metadata) == _R16_PORTS


@pytest.mark.parametrize("hazard", ("priority", "external", "cube"))
@pytest.mark.parametrize("deps", ((), ("a",)))
@pytest.mark.parametrize("phase", ("raw", "events", "stripped"))
def test_r16_root_data_waits_across_priority_external_ready_and_pipes(hazard, deps, phase):
    p = _add("p", ("late_input",) if hazard == "external" else ("analysis_input",))
    if hazard == "cube":
        p = replace(p, family="gemm", flops=1e8, vector_ops=0.0, hbm_bytes=0.0,
                    semantic_shape=(1, 1, 4), local_shape=(1, 1, 4),
                    padded_shape=(1, 1, 4), launch_shape=(1, 1, 4))
    c = _add("c", deps, metadata=(("control_deps", ("a", "p")),
                                 ("fusion_input_ports", _R16_PORTS)))
    priorities = {}

    class PriorityRecipe:
        def lower(self, context):
            program = KernelProgram(
                (Stage("service", context.work),),
                minimum_ms=context.minimum_ms)
            priority = getattr(context.op, "priority", 0)
            priorities[context.op.op_id] = priority
            return replace(program, stages=tuple(replace(stage, priority=priority)
                           for stage in program.stages))

    if hazard == "priority":
        p = _R16PriorityOp(**vars(replace(p, metadata=(("test_priority", 0),))))
        c = _R16PriorityOp(**vars(replace(c, metadata=(*c.metadata, ("test_priority", 100)))))
        assert c.priority > p.priority
    plan = canonical_plan(case(), (_add("a", ("analysis_input",)), p, c,
        replace(_add("tail", ("c",)), stream="aux")), options={
            "external_inputs": ("analysis_input", "late_input"),
            "external_ready_ms": {"late_input": 0.1}})
    if hazard == "priority":
        # The synthetic members are the declared `add` family; the physical
        # registry already binds that family, so override it explicitly.
        plan = replace(plan, book=RecipeBook(families={"add": PriorityRecipe()}))
    current = plan.rewrite(_r16_phase(plan.ops, phase))
    _, rows = _r16_ordered(current)
    if hazard == "priority":
        assert priorities["c"] == 100 and priorities["p"] == 0
    if hazard == "external":
        assert rows["p"]["ready_ms"] >= 0.1
    if hazard == "cube":
        assert current.graph.ops["p"].resource == "cube"
        assert current.graph.ops["c"].resource == "vector"
    assert c.deps == deps and input_port_entries(c.metadata) == _R16_PORTS


@pytest.mark.parametrize("members", (("p",), ("c",), ("p", "c")))
@pytest.mark.parametrize("phase", ("raw", "events", "stripped"))
def test_r16_stream_move_cannot_bypass_explicit_data_order(members, phase):
    c = _add("c", (), metadata=(("fusion_input_ports", _R16_PORTS),))
    plan = canonical_plan(case(), (_add("g", ("analysis_input",)),
                                  _add("p", ("g",)), c))
    moved = canonical_stream_move(plan, members, _R16_HARDWARE)
    clean = _strip_events(moved)
    chosen = {"raw": clean, "events": moved, "stripped": _strip_events(
        insert_stream_events(moved, _R16_HARDWARE))}[phase]
    if len(members) == 1:
        with pytest.raises(FeasibilityError, match="^graph_input_not_ordered:c:p$"):
            plan.rewrite(chosen)
        with pytest.raises(FeasibilityError, match="^graph_input_not_ordered:c:p$"):
            plan.evaluate(chosen)
    else:
        current = plan.rewrite(chosen)
        result, _ = _r16_ordered(current)
        assert plan.evaluate(chosen)["duration_ms"] == result["duration_ms"]
        assert current.graph.ops["c"].inputs == ("p:out", "p:out")
    assert plan.ops[-1] is c and c.deps == ()


@pytest.mark.parametrize("schedule", ("empty", "control", "direct", "transitive"))
def test_r16_execution_copies_preserve_ports_recipes_work_and_inputless_identity(
        monkeypatch, schedule):
    from mindie_sim.prediction import fusion_bridge
    from mindie_sim.prediction.cost import CostModel

    deps = {"empty": (), "control": ("g",), "direct": ("p",), "transitive": ("via",)}[schedule]
    c = _add("c", deps, metadata=(("control_deps", ("g", "via", "p")),
                                 ("fusion_input_ports", _R16_PORTS)))
    ops = (_add("g", ("analysis_input",)), _add("p", ("g",)),
           _add("via", ("p",)), c, _add("inputless"))
    plan = canonical_plan(case(), ops)
    original = fusion_bridge.graph_costs
    calls = []

    def observe(case_, execution_ops, **kwargs):
        calls.append(execution_ops)
        cost = CostModel(case_)
        for source, transient in zip(ops, execution_ops):
            assert replace(transient, deps=source.deps) == source
            assert transient.metadata is source.metadata
            assert full_operand_ports(transient) == full_operand_ports(source)
            assert plan.book.select(source) is plan.book.select(transient)
            assert cost.price(source, "none").work == cost.price(transient, "none").work
        return original(case_, execution_ops, **kwargs)

    monkeypatch.setattr(fusion_bridge, "graph_costs", observe)
    _r16_ordered(plan)
    assert len(calls) == 1 and calls[0][-1] is ops[-1]
    lowered = calls[0][3]
    if schedule in {"empty", "control"}:
        assert lowered is not c and lowered.deps == (*deps, "p")
    else:
        assert lowered is c  # No redundant direct edge over an existing path.
    assert plan.ops is ops and plan.ops[3] is c and c.deps == deps


@pytest.mark.parametrize("scheduled", (False, True))
@pytest.mark.parametrize("phase", ("raw", "events", "stripped"))
def test_r16_nonroot_data_survives_contraction_prefetch_and_events(scheduled, phase):
    c = _add("c", ("a",) if scheduled else (), metadata=(
        ("control_deps", ("a", "p")), ("fusion_input_ports", _R16_PORTS)))
    plan = canonical_plan(case(), (_add("a", ("analysis_input",)), _add("b", ("a",)),
        _add("g", ("analysis_input",), hbm=1e8), _add("p", ("g",)), c,
        replace(_add("tail", ("c",)), stream="aux")))
    decision = PrefetchDecision("c", (), 256, 256, 0, 256, 0,
                                1.0, 1e6, 0.0, True, "r16", ())
    for base in (plan, _contract(plan, ("a", "b"))):
        _r16_ordered(base.rewrite(_r16_phase(base.ops, phase)))
        pending, pid = add_prefetch(base, decision, _R16_HARDWARE)
        rewrite = settle_prefetch_rewrite(pending, decision, pid, 127, _R16_HARDWARE)
        settled = apply_prefetch_rewrite(pending, rewrite, pid)
        current = settled.rewrite(_r16_phase(settled.ops, phase))
        _r16_ordered(current)
        assert current.graph.ops["c"].inputs == ("p:out", "p:out", pid + ":out")
    assert c.deps == (("a",) if scheduled else ())
    assert input_port_entries(c.metadata) == _R16_PORTS


def _r17_phase(ops, phase):
    clean = _strip_events(ops)
    events = insert_stream_events(clean, _R16_HARDWARE)
    assert insert_stream_events(events, _R16_HARDWARE) == events
    return {"raw": clean, "events": events, "stripped": _strip_events(events)}[phase]


def _r17_effects(*effects):
    return (("gdn_state_effects", tuple(
        {"target": target, "version": version, "action": action}
        for target, version, action in effects)),)


def _r17_plan(kind, deps):
    metadata = (("control_deps", ("a", "p", "via")),)
    before = (_add("a", ("analysis_input",)),)
    if kind == "external":
        metadata += (("fusion_input_ports", (("late", "late:out"),) * 2),)
    else:
        before += (_add("g", ("late",) if kind == "delayed" else ("a",)),
                   _add("p", ("g",), metadata=_r17_effects(("cache", 1, "write"))))
        metadata += _r17_effects(("cache", 1, "read"))
    ops = (*before, _add("c", deps, metadata=metadata),
           replace(_add("tail", ("c",)), stream="aux"))
    return canonical_plan(case(), ops, options={
        "external_inputs": ("analysis_input", "late"),
        "external_ready_ms": {"late": 0.125}})


def _r17_causal(graph, result):
    assert graph.validate()
    rows = {row["op_id"]: row for row in result["timeline"]}
    for node in graph.ops.values():
        for value_id in node.inputs:
            producer = graph.values[value_id].producer
            if producer is not None:
                assert rows[node.op_id]["ready_ms"] >= rows[producer]["complete_ms"]
            elif value_id == "late:source":
                assert rows[node.op_id]["ready_ms"] >= 0.125
    return rows


@pytest.mark.parametrize("kind", ("external", "state", "delayed"))
@pytest.mark.parametrize("schedule", ("empty", "unrelated", "direct"))
@pytest.mark.parametrize("capture", (None, "full"))
@pytest.mark.parametrize("phase", ("raw", "events", "stripped"))
def test_r17_external_and_state_readiness_both_entry_points(kind, schedule, capture, phase):
    deps = {"empty": (), "unrelated": ("a",),
            "direct": ("late",) if kind == "external" else ("p",)}[schedule]
    base = _r17_plan(kind, deps)
    chosen = _r17_phase(base.ops, phase)
    current = base.rewrite(chosen)
    inputs = (("late:source",) * 2 if kind == "external" else ("p:state_out:cache@1",))
    assert current.graph.ops["c"].inputs == inputs
    saved = tuple((op, op.deps, op.metadata, repr(op.metadata)) for op in chosen)
    ordinary = current.evaluate(capture_mode=capture)
    override = base.evaluate(chosen, capture_mode=capture)
    rows = _r17_causal(current.graph, ordinary)
    assert rows == _r17_causal(current.graph, override)
    assert ordinary["duration_ms"] == override["duration_ms"]
    assert ordinary["resources"] == override["resources"]
    if kind == "external":
        assert rows["c"]["ready_ms"] == 0.125
        assert ordinary["duration_ms"] == pytest.approx(0.14 if phase == "events" else 0.13)
    for op, original_deps, metadata, representation in saved:
        assert op.deps is original_deps and op.metadata is metadata
        assert repr(metadata) == representation
    assert next(op for op in base.ops if op.op_id == "c").deps == deps


@pytest.mark.parametrize("members", (("p",), ("c",), ("p", "c")))
@pytest.mark.parametrize("schedule", ("empty", "direct", "transitive"))
@pytest.mark.parametrize("capture", (None, "full"))
@pytest.mark.parametrize("phase", ("raw", "events", "stripped"))
def test_r17_state_partial_comoving_and_synchronized_streams(
        members, schedule, capture, phase):
    base = _r17_plan("delayed", ())
    reader = replace(base.ops[-2], deps={"empty": (), "direct": ("p",),
                                       "transitive": ("via",)}[schedule])
    ops = (*base.ops[:-2], _add("via", ("p",)), reader, base.ops[-1])
    base = base.rewrite(ops)
    # The public optimizer still treats state effects as stream-move barriers.
    with pytest.raises(ValueError, match="stream_move_preserves_barrier"):
        canonical_stream_move(base, members, _R16_HARDWARE)
    # User-authored selected tuples must obey the same crossing contract.
    moved = tuple(replace(op, stream="aux") if op.op_id in members else op for op in ops)
    chosen = _r17_phase(moved, phase)
    if schedule == "empty" and len(members) == 1:
        with pytest.raises(FeasibilityError, match="^graph_input_not_ordered:c:p$"):
            base.rewrite(chosen)
        for current, override in ((base, chosen), (replace(base, ops=chosen), None)):
            with pytest.raises(FeasibilityError, match="^graph_input_not_ordered:c:p$"):
                current.evaluate(override, capture_mode=capture)
    else:
        current = base.rewrite(chosen)
        assert current.graph.ops["c"].inputs == ("p:state_out:cache@1",)
        _r17_causal(current.graph, current.evaluate(capture_mode=capture))
        _r17_causal(current.graph, base.evaluate(chosen, capture_mode=capture))
        if phase == "events" and schedule != "empty" and members == ("p",):
            assert any(op.family == "event" for op in chosen)
    assert base.ops is ops


@pytest.mark.parametrize("member", ("p", "c"))
def test_r17_unscheduled_state_stage_crossing_rejected_before_execution(monkeypatch, member):
    from mindie_sim.prediction import fusion_bridge
    base = _r17_plan("state", ())
    chosen = tuple(replace(op, stage=1) if op.op_id == member else op for op in base.ops)

    def forbidden(*args, **kwargs):
        pytest.fail("invalid state crossing reached graph_costs")

    monkeypatch.setattr(fusion_bridge, "graph_costs", forbidden)
    with pytest.raises(FeasibilityError, match="^graph_input_not_ordered:c:p$"):
        base.rewrite(chosen)
    for current, override in ((base, chosen), (replace(base, ops=chosen), None)):
        with pytest.raises(FeasibilityError, match="^graph_input_not_ordered:c:p$"):
            current.evaluate(override)


@pytest.mark.parametrize("kind", ("external", "state"))
@pytest.mark.parametrize("fusion", (False, True))
@pytest.mark.parametrize("prefetch", ("none", "pending", "settled"))
@pytest.mark.parametrize("move", (False, True))
@pytest.mark.parametrize("capture", (None, "full"))
@pytest.mark.parametrize("phase", ("raw", "events", "stripped"))
def test_r17_declared_boundaries_compose_with_fusion_prefetch_streams(
        kind, fusion, prefetch, move, capture, phase):
    base = _r17_plan(kind, ())
    original_reader = base.ops[-2]
    ops = (*base.ops[:1], _add("b", ("a",)), *base.ops[1:-1],
           _add("d", ("c",)), replace(base.ops[-1], deps=("d",)))
    if kind == "state":
        # The writer's gate is itself explicit external data absent from deps.
        ops = tuple(replace(op, deps=(), metadata=(
            ("fusion_input_ports", (("late", "late:out"),) * 2),))
            if op.op_id == "g" else op for op in ops)
    base = base.rewrite(ops)
    if move:
        base = base.rewrite(canonical_stream_move(
            base, ("c", "d") if kind == "external" else ("g",), _R16_HARDWARE))
    target = "c"
    if fusion:
        base = _contract(base, ("c", "d") if kind == "external" else ("a", "b"))
        target = "fused:c+d" if kind == "external" else "c"
    if prefetch != "none":
        decision = PrefetchDecision(target, (), 256, 256, 0, 256, 0,
                                    1.0, 1e6, 0.0, True, "r17", ())
        base, pid = add_prefetch(base, decision, _R16_HARDWARE)
        if prefetch == "settled":
            rewrite = settle_prefetch_rewrite(base, decision, pid, 127, _R16_HARDWARE)
            base = apply_prefetch_rewrite(base, rewrite, pid)
    chosen = _r17_phase(base.ops, phase)
    current = base.rewrite(chosen)
    for result in (current.evaluate(capture_mode=capture),
                   base.evaluate(chosen, capture_mode=capture)):
        rows = _r17_causal(current.graph, result)
        assert rows[target]["ready_ms"] >= 0.125
    inputs = current.graph.ops[target].inputs
    if kind == "external":
        # A fused kernel's physical boundary binds an exact duplicate once.
        expected = ("late:source",) * (1 if fusion else 2)
        assert inputs == (*expected, *((pid + ":out",) if prefetch == "settled" else ()))
        assert input_port_entries(original_reader.metadata) == (("late", "late:out"),) * 2
    else:
        assert "p:state_out:cache@1" in inputs
    assert original_reader.deps == ()


@pytest.mark.parametrize("scheduled", (False, True))
@pytest.mark.parametrize("stream", ("main", "aux"))
@pytest.mark.parametrize("capture", (None, "full"))
@pytest.mark.parametrize("phase", ("raw", "events", "stripped"))
def test_r17_multi_target_state_read_modify_write_chain(scheduled, stream, capture, phase):
    ops = (_add("p", ("late",), metadata=_r17_effects(("cache", 7, "write"))),
           _add("q", ("late",), metadata=_r17_effects(("stats", 2, "write"))))
    for name, deps, a, b in (("c", ("p", "q"), 7, 2), ("m", ("c",), 8, 3)):
        ops += (_add(name, deps if scheduled else (), metadata=(
            ("control_deps", deps), *_r17_effects(
                ("cache", a, "read"), ("cache", a + 1, "write"),
                ("stats", b, "read"), ("stats", b + 1, "write")))),)
    ops += (_add("r", ("m",) if scheduled else (), metadata=(
        ("control_deps", ("m",)), *_r17_effects(
            ("cache", 9, "read"), ("stats", 4, "read")))),)
    ops = tuple(replace(op, stream=stream) for op in ops) + (
        replace(_add("tail", ("r",)), stream="aux" if stream == "main" else "main"),)
    base = canonical_plan(case(), ops, options={
        "external_inputs": ("late",), "external_ready_ms": {"late": 0.125}})
    chosen = _r17_phase(ops, phase)
    current = base.rewrite(chosen)
    assert current.graph.ops["c"].inputs == ("p:state_out:cache@7", "q:state_out:stats@2")
    assert current.graph.ops["m"].inputs == ("c:state_out:cache@8", "c:state_out:stats@3")
    assert current.graph.ops["r"].inputs == ("m:state_out:cache@9", "m:state_out:stats@4")
    for result in (current.evaluate(capture_mode=capture),
                   base.evaluate(chosen, capture_mode=capture)):
        _r17_causal(current.graph, result)


@pytest.mark.parametrize("container", (tuple, list, dict, UserDict, MappingProxyType))
@pytest.mark.parametrize("kind", ("external", "state", "mixed"))
@pytest.mark.parametrize("schedule", ("empty", "unrelated", "direct", "transitive"))
def test_r17_lowering_identity_external_union_and_no_redundant_edges(
        monkeypatch, container, kind, schedule):
    from mindie_sim.prediction import fusion_bridge
    from mindie_sim.prediction.cost import CostModel
    direct = "late" if kind == "external" else "p"
    deps = {"empty": (), "unrelated": ("a",), "direct": (direct,),
            "transitive": ("via",)}[schedule]
    metadata = (("control_deps", ("a", "p", "via")),)
    if kind != "external":
        metadata += _r17_effects(("cache", 1, "read"))
    if kind != "state":
        metadata += (("fusion_input_ports", (("late", "late:out"),) * 2),)
    c = replace(_add("c", deps), metadata=container(
        metadata if container in (tuple, list) else dict(metadata)))
    ops = (_add("a", ("analysis_input",)), _add("p", (), metadata=(
        ("fusion_input_ports", (("late", "late:out"),)),
        *_r17_effects(("cache", 1, "write")))), _add("via", ("p",)), c,
        _add("inputless"), _add("pure_control", (), metadata=(("control_deps", ("late",)),)))
    base = canonical_plan(case(), (_add("old", ("analysis_input",)),), options={
        "external_inputs": ("analysis_input", "late"),
        "external_ready_ms": {"late": 0.125}})
    assert "late" not in base.external_inputs
    current = base.rewrite(ops)
    graph = current.graph
    seen = []

    def observe(case_, lowered, **kwargs):
        seen.append(lowered)
        assert kwargs["external_inputs"]["late"] == 0.125
        model = CostModel(case_)
        for source, transient in zip(ops, lowered):
            assert replace(transient, deps=source.deps) == source
            assert transient.metadata is source.metadata
            assert full_operand_ports(transient) == full_operand_ports(source)
            assert model.price(source, "none").work == model.price(transient, "none").work
            assert kwargs["recipes"].select(source) is kwargs["recipes"].select(transient)
        assert lowered[0] is ops[0] and lowered[2] is ops[2]
        assert lowered[1].deps == ("late",)
        assert lowered[4] is ops[4] and lowered[5] is ops[5]
        expected = (*deps, direct) if schedule in {"empty", "unrelated"} else deps
        assert lowered[3].deps == expected
        assert (lowered[3] is c) is (expected == deps)
        return {"observed": True}

    monkeypatch.setattr(fusion_bridge, "graph_costs", observe)
    assert current.evaluate() == base.evaluate(ops) == {"observed": True}
    assert replace(base, ops=ops).evaluate() == {"observed": True}
    assert len(seen) == 3 and current.graph is graph and current.ops is ops
    assert c.deps is deps
