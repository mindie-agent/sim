"""NR-S2-FIX9/FIX19/FIX20: architectural and pool rollback witnesses.

[Role] Reproduce the two remaining Round-9 root causes: an implicit caller
  event promoted to a data operand by an event rewrite, and many-peer
  progressive-fill accumulation that either falsely raises or silently exceeds
  the 16-ULP unit bound. FIX19 covers all six timeout propagation/cleanup sites;
  FIX20 covers token-only release rollback, generation ordering and clean reuse.
[Invariants] Data and control dependencies are classified by one canonical
  rule across projection and port materialization, so a caller event stays a
  scheduling edge through move/insert/reinsert/strip/contraction.  Every
  per-unit committed total is an exact ``math.fsum`` of the returned
  ``alpha * weight`` terms, each task cap and coupled lockstep is preserved,
  and the actual sum never exceeds the capacity plus 16 ULP.
[Timeouts & Locks] Pure in-process simulation; no clocks, sockets or locks.
[Side-Effect Policy] No filesystem, environment or device mutation.
"""
from __future__ import annotations

from dataclasses import replace
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mindie_sim.prediction import canonical_optimizer, engine_plan, fusion_bridge
from mindie_sim.prediction.fusion_bridge import (
    _strip_events, apply_rewrite, canonical_plan, canonical_stream_move,
    insert_stream_events, rewrite_fusion,
)
from mindie_sim.prediction.fusion_recipe import (
    RejectedFusion, synthesize_vector,
)
from mindie_sim.prediction.fusion_pool import (
    PoolHolder, PrefetchPoolProgram, ReleasePoolProgram,
)
from mindie_sim.prediction.plan_model import PlanNode
from mindie_sim.prediction.runtime.engine import Engine, Resource

from graph_opt_fixtures import hw
from test_graph_optimizer_r5_adversarial import _busy_prefetch_plan
from test_graph_optimizer_r8_adversarial import _add, case

_ULP_LIMIT = 16.0


def _caller_event(op_id="event:caller", dep="a"):
    return replace(
        _add(op_id, (dep,), family="event",
             metadata=(("event_kind", "caller"),)),
        stream="main",
    )


def _reachable(ops, start):
    by_id = {op.op_id: op for op in ops}
    seen: set[str] = set()
    pending = list(by_id[start].deps)
    while pending:
        dep = pending.pop()
        if dep in seen:
            continue
        seen.add(dep)
        node = by_id.get(dep)
        if node is not None:
            pending.extend(node.deps)
    return seen


def _plan_and_rewrites(*ops):
    plan = canonical_plan(case(), tuple(ops))
    hardware = hw(max_streams=2, stream_switch_ms=0.01)
    moved = canonical_stream_move(plan, ("c",), hardware)
    once = insert_stream_events(moved, hardware)
    twice = insert_stream_events(once, hardware)
    stripped = _strip_events(moved)
    return plan, (moved, once, twice, stripped)


def _consumer(rewritten, op_id="c"):
    return next(op for op in rewritten if op.op_id == op_id)


# ---------------------------------------------------------------------------
# Finding A: an implicit caller event is scheduling-only, never an operand
# ---------------------------------------------------------------------------
def test_r9_fa_implicit_caller_event_stays_control_across_rewrites():
    plan, rewrites = _plan_and_rewrites(
        _add("a", ("analysis_input",)),
        _caller_event(),
        _add("c", ("a", "event:caller", "a")),
    )
    expected = ("a:out", "a:out")
    assert plan.graph.ops["c"].inputs == expected
    for rewritten in rewrites:
        after = plan.rewrite(rewritten)
        assert after.graph.validate()
        node = after.graph.ops["c"]
        assert node.inputs == expected
        assert "event:caller:out" not in node.inputs
        metadata = dict(_consumer(rewritten).metadata)
        assert metadata["fusion_input_ports"] == (
            ("a", "a:out"), ("a", "a:out"))
        assert any(op.op_id == "event:caller" for op in rewritten)
        assert "event:caller" in _reachable(rewritten, "c")
    stripped = _consumer(rewrites[-1])
    assert stripped.deps == ("a", "event:caller")


def test_r9_fa_named_event_family_without_prefix_stays_control():
    plan, rewrites = _plan_and_rewrites(
        _add("a", ("analysis_input",)),
        _caller_event("caller_gate"),
        _add("c", ("a", "caller_gate", "a")),
    )
    assert plan.graph.ops["c"].inputs == ("a:out", "a:out")
    for rewritten in rewrites:
        after = plan.rewrite(rewritten)
        assert after.graph.validate()
        assert after.graph.ops["c"].inputs == ("a:out", "a:out")
        assert any(op.op_id == "caller_gate" for op in rewritten)
        assert "caller_gate" in _reachable(rewritten, "c")
    assert _consumer(rewrites[-1]).deps == ("a", "caller_gate")


def test_r9_fa_explicit_control_dep_and_explicit_data_port_controls():
    hardware = hw(max_streams=2, stream_switch_ms=0.01)
    # Positive control 1: an explicit control declaration is honoured and the
    # same classification still materializes only the two data operands.
    control_plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _caller_event(),
        _add("c", ("a", "event:caller", "a"),
             metadata=(("control_deps", ("event:caller",)),)),
    ))
    assert control_plan.graph.ops["c"].inputs == ("a:out", "a:out")
    moved = canonical_stream_move(control_plan, ("c",), hardware)
    for rewritten in (moved, insert_stream_events(moved, hardware),
                      _strip_events(moved)):
        after = control_plan.rewrite(rewritten)
        after.graph.validate()
        assert after.graph.ops["c"].inputs == ("a:out", "a:out")
        assert "event:caller" in _reachable(rewritten, "c")

    # Positive control 2: an explicit data-port binding for the caller event is
    # still caller-authoritative; the classifier never forbids the event node.
    entries = (("a", "a:out"), ("event:caller", "event:caller:out"),
               ("a", "a:out"))
    data_plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _caller_event(),
        _add("c", ("a", "event:caller", "a"),
             metadata=(("fusion_input_ports", entries),)),
    ))
    expected = ("a:out", "event:caller:out", "a:out")
    assert data_plan.graph.ops["c"].inputs == expected
    moved = canonical_stream_move(data_plan, ("c",), hardware)
    for rewritten in (moved, insert_stream_events(moved, hardware),
                      _strip_events(moved)):
        after = data_plan.rewrite(rewritten)
        after.graph.validate()
        assert after.graph.ops["c"].inputs == expected
        assert any(op.op_id == "event:caller" for op in rewritten)


def test_r9_fa_partial_explicit_binding_excludes_implicit_event():
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _caller_event(),
        _add("c", ("a", "event:caller", "a"),
             metadata=(("fusion_input_ports", (("a", "a:out"),)),)),
    ))
    assert plan.graph.ops["c"].inputs == ("a:out", "a:out")
    hardware = hw(max_streams=2, stream_switch_ms=0.01)
    moved = canonical_stream_move(plan, ("c",), hardware)
    for rewritten in (moved, insert_stream_events(moved, hardware),
                      _strip_events(moved)):
        after = plan.rewrite(rewritten)
        after.graph.validate()
        assert after.graph.ops["c"].inputs == ("a:out", "a:out")
        assert "event:caller" in _reachable(rewritten, "c")
        metadata = dict(_consumer(rewritten).metadata)
        assert metadata["fusion_input_ports"] == (
            ("a", "a:out"), ("a", "a:out"))


def test_r9_fa_contraction_preserves_control_and_data_arity():
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _caller_event(),
        _add("b", ("a",), family="rotary",
             metadata=(("rotate_style", "interleaved"),)),
        _add("c", ("a", "event:caller", "a")),
    ))
    candidate = synthesize_vector(
        plan.graph, ("a", "b"), tile_rows=1,
        capacity_per_owner=plan.hardware.ub_bytes_per_aiv)
    assert not isinstance(candidate, RejectedFusion), candidate.reasons
    updated = apply_rewrite(plan, rewrite_fusion(plan, candidate))
    graph = updated.graph
    graph.validate()
    node = graph.ops["c"]
    fused_id = "fused:a+b"
    assert len(node.inputs) == 2
    assert node.inputs[0] == node.inputs[1]
    assert graph.values[node.inputs[0]].producer == fused_id
    assert "event:caller" in node.deps
    assert "event:caller:out" not in node.inputs
    assert graph.ops["event:caller"].deps == (fused_id,)


def test_r9_fa_fused_region_keeps_member_control_dependency():
    plan = canonical_plan(case(), (
        _add("a", ("analysis_input",)),
        _caller_event(),
        _add("b", ("a",), family="rotary",
             metadata=(("rotate_style", "interleaved"),)),
        _add("c", ("a", "b", "event:caller")),
        _add("d", ("c",)),
    ))
    candidate = synthesize_vector(
        plan.graph, ("c", "d"), tile_rows=1,
        capacity_per_owner=plan.hardware.ub_bytes_per_aiv)
    assert not isinstance(candidate, RejectedFusion), candidate.reasons
    rewrite = rewrite_fusion(plan, candidate)
    updated = apply_rewrite(plan, rewrite)
    updated.graph.validate()
    assert "event:caller" in rewrite.fused.deps
    assert updated.graph.ops[rewrite.fused.op_id].deps == rewrite.fused.deps
    assert "event:caller" in _reachable(updated.ops, rewrite.fused.op_id)
    assert all("event:caller:out" not in node.inputs
               for node in updated.graph.ops.values())


# ---------------------------------------------------------------------------
# Finding B: exact fsum per-unit accounting for many simultaneous peers
# ---------------------------------------------------------------------------
def _peer_engine(capacity, names=("hbm", "vector")):
    return Engine([Resource(name, capacity,
                            "bytes" if name == "hbm" else "ops")
                   for name in names])


def _activate(engine):
    engine.advance(horizon=0.0, first=False)
    return engine._rates()


def _assert_unit_bounds(rates, capacities):
    for name, capacity in capacities.items():
        rows = rates.get(name, {})
        total = math.fsum(rows.values())
        ulp = math.ulp(capacity)
        excess_ulps = max(0.0, (total - capacity) / ulp)
        assert 0.0 <= excess_ulps <= _ULP_LIMIT, (
            name, total, capacity, excess_ulps)
        assert total <= capacity + _ULP_LIMIT * ulp
        assert all(math.isfinite(rate) and rate >= 0.0
                   for rate in rows.values())


def _identical_peers(count, capacity):
    engine = _peer_engine(capacity)
    events = tuple(engine.submit(
        str(index), {"hbm": capacity, "vector": capacity}, coupled=True)
        for index in range(count))
    return engine, events


def test_r9_fb_hundred_peer_exact_witness_does_not_false_raise():
    capacity = 4121231867897.778
    engine, events = _identical_peers(100, capacity)
    rates = _activate(engine)
    assert len(rates["hbm"]) == len(rates["vector"]) == 100
    _assert_unit_bounds(rates, {"hbm": capacity, "vector": capacity})
    engine.advance(until=events, first=False, max_steps=10)
    assert all(event.done for event in events)
    assert max(event.time for event in events) == pytest.approx(100.0, rel=1e-9)
    assert engine.summary()["live_tasks"] == 0


@pytest.mark.parametrize("capacity", (8332484570.630881, 125548680209.0561))
def test_r9_fb_three_hundred_peer_exact_witnesses(capacity):
    engine, events = _identical_peers(300, capacity)
    rates = _activate(engine)
    assert len(rates["hbm"]) == len(rates["vector"]) == 300
    _assert_unit_bounds(rates, {"hbm": capacity, "vector": capacity})
    engine.advance(until=events, first=False, max_steps=10)
    assert all(event.done for event in events)
    assert max(event.time for event in events) == pytest.approx(300.0, rel=1e-9)
    assert engine.summary()["live_tasks"] == 0


def test_r9_fb_thousand_peer_exact_bound_and_completion():
    capacity = 4121231867897.778
    engine, events = _identical_peers(1000, capacity)
    rates = _activate(engine)
    assert len(rates["hbm"]) == len(rates["vector"]) == 1000
    _assert_unit_bounds(rates, {"hbm": capacity, "vector": capacity})
    engine.advance(until=events, first=False, max_steps=10)
    assert all(event.done for event in events)
    assert max(event.time for event in events) == pytest.approx(1000.0, rel=1e-9)


def test_r9_fb_five_hundred_peer_mixed_cap_coupled_fsum():
    capacity = 125548680209.0561
    engine = _peer_engine(capacity, ("hbm", "vector", "mte2"))
    events = []
    caps_by_sequence: dict[int, tuple[str, float]] = {}
    coupled_sequences: set[int] = set()
    for index in range(300):
        caps = {"hbm": capacity / 1000.0} if index % 7 == 0 else None
        event = engine.submit(
            f"c{index}", {"hbm": capacity, "vector": capacity},
            coupled=True, caps=caps)
        events.append(event)
        coupled_sequences.add(event.sequence)
        if caps:
            caps_by_sequence[event.sequence] = ("hbm", caps["hbm"])
    for index in range(200):
        name = ("hbm", "vector", "mte2")[index % 3]
        caps = {name: capacity / 1000.0} if index % 11 == 0 else None
        event = engine.submit(f"u{index}", {name: capacity}, caps=caps)
        events.append(event)
        if caps:
            caps_by_sequence[event.sequence] = (name, caps[name])
    rates = _activate(engine)
    assert set(rates) == {"hbm", "vector", "mte2"}
    assert all(rates[name] for name in rates)
    _assert_unit_bounds(
        rates, {"hbm": capacity, "vector": capacity, "mte2": capacity})
    for sequence, (name, cap) in caps_by_sequence.items():
        rate = rates[name].get(sequence, 0.0)
        assert rate > 0.0
        assert rate <= cap + _ULP_LIMIT * math.ulp(cap)
    for sequence in coupled_sequences:
        values = [rows[sequence] for rows in rates.values()
                  if sequence in rows]
        assert len(values) == 2
        fractions = [value / capacity for value in values]
        assert max(fractions) - min(fractions) <= 1e-9 * max(1.0, fractions[0])
    engine.advance(until=events, first=False, max_steps=100000)
    assert all(event.done for event in events)
    assert engine.summary()["live_tasks"] == 0
    assert engine.now == pytest.approx(1000.0, rel=1e-9)


class _DeadlineExpired(TimeoutError):
    """Timeout subclasses must propagate as the same exception object too."""


@pytest.mark.parametrize("error_type", (TimeoutError, _DeadlineExpired))
@pytest.mark.parametrize("stage", ("trial", "settled"))
def test_fix19_canonical_prefetch_timeout_is_immediate(monkeypatch, stage, error_type):
    pending, _, _ = _busy_prefetch_plan()
    # The predecessor must leave spare HBM bandwidth in the consumer's ready
    # window; an HBM-exact predecessor makes the real prefetch planner reject
    # with ``hbm_saturated_no_overlap_gain`` before either evaluation stage.
    plan = pending.rewrite(tuple(
        replace(op, hbm_bytes=16 * 1024 * 1024) if op.op_id == "mm"
        else replace(op, vector_ops=1e7) if op.op_id == "pre" else op
        for op in pending.ops
        if op.family not in {"prefetch", "release"} and op.op_id != "busy"))
    sentinel = error_type(stage)
    calls = []
    evaluate = fusion_bridge.CanonicalPlan.evaluate
    budget = canonical_optimizer.Budget(per_pass=32, total=64, wall_s=30.0)

    def observe(self, *args, **kwargs):
        current = "audit" if kwargs.get("audit_allocations") else "baseline"
        if any(op.family == "prefetch" for op in self.ops):
            current = "settled" if any(dict(op.metadata).get("prefetch_hit_bytes")
                                       for op in self.ops) else "trial"
        calls.append(current)
        if current == stage:
            raise sentinel
        return evaluate(self, *args, **kwargs)

    monkeypatch.setattr(fusion_bridge.CanonicalPlan, "evaluate", observe)
    monkeypatch.setattr(canonical_optimizer, "_candidate_specs", lambda *a, **k: ((), ()))
    monkeypatch.setattr(canonical_optimizer, "Budget", lambda **kwargs: budget)
    with pytest.raises(error_type) as caught:
        canonical_optimizer._run_canonical(
            plan, evaluation_budget=64, per_pass=32, max_streams=None, wall_budget_s=30.0)
    assert caught.value is sentinel
    assert calls == ["baseline", "audit", "trial"] + (["settled"] if stage == "settled" else [])
    assert budget.used_total == len(calls)  # The failing evaluation still charges once.


@pytest.mark.parametrize("error_type", (TimeoutError, _DeadlineExpired))
def test_fix19_opaque_ids_timeout_is_immediate(error_type):
    sentinel = error_type("recipe selection")
    book = SimpleNamespace(select=Mock(side_effect=sentinel))
    op = SimpleNamespace(source_boundary=False, op_id="candidate")
    with pytest.raises(error_type) as caught:
        fusion_bridge._opaque_ids((op, op), book)
    assert caught.value is sentinel
    book.select.assert_called_once_with(op)


@pytest.mark.parametrize("error_type", (TimeoutError, _DeadlineExpired))
def test_fix19_evaluate_plan_timeout_is_immediate(monkeypatch, error_type):
    sentinel = error_type("engine construction")
    construct = Mock(side_effect=sentinel)
    monkeypatch.setattr(engine_plan, "Engine", construct)
    with pytest.raises(error_type) as caught:
        engine_plan.evaluate_plan((PlanNode("one", work=(("vector", 1.0),)),), hw())
    assert caught.value is sentinel
    assert construct.call_count == 1


def _fix19_pool():
    engine = Engine([Resource("hbm", 1024, "bytes")],
                    pools={"l1": 4096}, pool_units={"l1": 1})
    ready = engine.submit("entry")
    holder = PoolHolder("fix19")
    peer = PrefetchPoolProgram(holder, 1024, 1).start(engine, "peer", ready)
    return engine, ready, holder, peer


@pytest.mark.parametrize("error_type", (TimeoutError, RuntimeError, BaseException))
@pytest.mark.parametrize("amount", (0, 1024))
def test_fix19_failed_copy_retires_only_its_allocation(monkeypatch, error_type, amount):
    engine, ready, holder, peer = _fix19_pool()
    bucket = holder._bucket(engine)
    peer_grant = bucket.by_id[bucket.capabilities[("handle", peer.sequence)]]
    sentinel = error_type("copy submission")
    retire = Mock(wraps=holder.retire)
    monkeypatch.setattr(holder, "retire", retire)
    submit = Mock(side_effect=sentinel)
    with monkeypatch.context() as injected:
        injected.setattr(engine, "submit", submit)
        with pytest.raises(error_type) as caught:
            PrefetchPoolProgram(holder, amount, 1).start(engine, "copy", ready)
    assert caught.value is sentinel and submit.call_count == 1
    assert list(bucket.active_by_key.values()) == [peer_grant]
    assert not peer_grant.retired and not peer_grant.release_scheduled
    assert bucket.copy_handles == {peer.sequence}
    if amount:
        failed = bucket.by_id[f"grant:{holder.generation}"]
        assert failed.retired and failed.allocation is not peer_grant.allocation
        retire.assert_called_once_with(engine, failed.allocation)
    else:
        retire.assert_not_called()
    # The failed key is reusable; successful submission must keep the new grant.
    copied = PrefetchPoolProgram(holder, 1024, 1).start(engine, "copy", ready)
    active = bucket.by_id[bucket.capabilities[("handle", copied.sequence)]]
    assert not active.retired and bucket.active_by_key[active.key] is active


@pytest.mark.parametrize("error_type", (TimeoutError, RuntimeError, BaseException))
def test_fix19_failed_release_resets_claim_and_can_retry(monkeypatch, error_type):
    engine, ready, holder, peer = _fix19_pool()
    copied = PrefetchPoolProgram(holder, 1024, 1).start(engine, "copy", ready)
    bucket = holder._bucket(engine)
    grant = bucket.by_id[bucket.capabilities[("handle", copied.sequence)]]
    peer_grant = bucket.by_id[bucket.capabilities[("handle", peer.sequence)]]
    sentinel = error_type("free submission")
    submit = Mock(side_effect=sentinel)
    complete = Mock(wraps=engine.on_complete)
    monkeypatch.setattr(engine, "on_complete", complete)
    release = ReleasePoolProgram(holder)
    with monkeypatch.context() as injected:
        injected.setattr(engine, "submit", submit)
        with pytest.raises(error_type) as caught:
            release.start(engine, "copy:release", copied, copy_handle=copied)
    assert caught.value is sentinel and submit.call_count == 1
    assert not grant.release_scheduled and not grant.retired
    assert bucket.active_by_key[grant.key] is grant
    assert not peer_grant.retired and not peer_grant.release_scheduled
    complete.assert_not_called()
    freed = release.start(engine, "copy:release", copied, copy_handle=copied)
    assert grant.release_scheduled and not grant.retired
    assert complete.call_count == 1
    engine.advance(until=(freed,), first=False)
    assert grant.retired and grant.key not in bucket.active_by_key
    assert bucket.active_by_key[peer_grant.key] is peer_grant
    assert engine.summary()["live_allocated"] == {"l1": 1024.0}


def _fix20_pool(units):
    engine = Engine([Resource("hbm", 1536, "bytes")],
                    pools={"l1": 8192}, pool_units={"l1": units})
    entry = engine.submit("entry")
    holder = PoolHolder("fix20")
    peer = PrefetchPoolProgram(holder, 768, units, lease_token="peer").start(
        engine, "peer", entry)
    engine.advance(until=(peer,), first=False)
    return engine, entry, holder, peer, holder._bucket(engine)


def _fix20_grant(bucket, event):
    return bucket.by_id[bucket.capabilities[("handle", event.sequence)]]


@pytest.mark.parametrize("error_type", (TimeoutError, _DeadlineExpired, RuntimeError, BaseException))
@pytest.mark.parametrize("units", (1, 3))
@pytest.mark.parametrize("retry_mode", ("token", "explicit_handle", "implicit_handle"))
def test_fix20_token_failure_exact_retry_retirement_and_reuse(
        monkeypatch, error_type, units, retry_mode):
    engine, entry, holder, peer, bucket = _fix20_pool(units)
    copy = PrefetchPoolProgram(holder, 1536, units, lease_token="T")
    copied = copy.start(engine, "copy", entry)
    engine.advance(until=(copied,), first=False)
    grant, peer_grant = _fix20_grant(bucket, copied), _fix20_grant(bucket, peer)
    assert grant.allocation.units == tuple(range(units))
    assert grant.allocation.size * units == 1536
    consumer = engine.submit("consumer", after=(copied,), latency=0.5)
    queue = {k: list(v) for k, v in bucket.token_live.items()}
    capabilities = dict(bucket.capabilities)
    sentinel = error_type("token free submission")
    submit = Mock(side_effect=sentinel)
    complete = Mock(wraps=engine.on_complete)
    release = ReleasePoolProgram(holder, lease_token="T")
    with monkeypatch.context() as injected:
        injected.setattr(engine, "submit", submit)
        injected.setattr(engine, "on_complete", complete)
        with pytest.raises(error_type) as caught:
            release.start(engine, "copy:release", consumer)
    assert caught.value is sentinel and submit.call_count == 1
    assert submit.call_args.args == ("copy:release:free",)
    assert submit.call_args.kwargs["free"] == (grant.allocation,)
    complete.assert_not_called()
    assert not grant.release_scheduled and not grant.retired
    assert bucket.active_by_key[grant.key] is grant
    assert bucket.token_live == queue and bucket.capabilities == capabilities
    assert engine.summary()["live_allocated"] == {"l1": 2304.0}
    with pytest.raises(ValueError, match="pool_lease_key_already_live"):
        copy.start(engine, "copy", entry)
    with pytest.raises(ValueError, match="pool_lease_token_already_live"):
        copy.start(engine, "r0:i1:copy", entry)
    ready = copied if retry_mode == "implicit_handle" else consumer
    kwargs = {"copy_handle": copied} if retry_mode == "explicit_handle" else {}
    freed = release.start(engine, "copy:release", ready, **kwargs)
    assert freed.name == "copy:release:free" and grant.release_scheduled
    assert not grant.retired and bucket.active_by_key[grant.key] is grant
    assert bucket.token_live == {"peer": [peer_grant.grant_id]}
    with pytest.raises(ValueError, match="pool_lease_key_already_live"):
        copy.start(engine, "copy", entry)
    if retry_mode != "implicit_handle":
        engine.advance(horizon=engine.now + 0.25, first=False)
        assert not freed.done and not grant.retired
        assert engine.summary()["live_allocated"] == {"l1": 2304.0}
    engine.advance(until=(freed,), first=False)
    assert freed.done and grant.retired and grant.key not in bucket.active_by_key
    if retry_mode != "implicit_handle":
        assert freed.time == consumer.time
    assert bucket.active_by_key[peer_grant.key] is peer_grant
    assert not peer_grant.retired and not peer_grant.release_scheduled
    assert engine.summary()["live_allocated"] == {"l1": 768.0}
    reused = copy.start(engine, "copy", entry)
    newer = _fix20_grant(bucket, reused)
    assert newer.grant_id != grant.grant_id and newer.key == grant.key
    assert bucket.capabilities[("token", "T")] == grant.grant_id
    for kwargs in ({}, {"copy_handle": copied}):
        stale = release.start(engine, "copy:release", copied, **kwargs)
        assert stale.name == "copy:release:noop"
        assert not newer.retired and not newer.release_scheduled
    done = engine.submit("reused-consumer", after=(reused, consumer))
    final = release.start(engine, "copy:release", done)
    peer_free = ReleasePoolProgram(holder).start(engine, "peer:release", peer)
    engine.advance(until=(final, peer_free), first=False)
    assert newer.retired and not bucket.active_by_key and not bucket.token_live
    assert engine.summary()["live_allocated"] == {"l1": 0.0}
    assert engine.summary()["live_tasks"] == 0


@pytest.mark.parametrize("error_type", (TimeoutError, _DeadlineExpired, RuntimeError, BaseException))
@pytest.mark.parametrize("units", (1, 3))
@pytest.mark.parametrize("committed_prefix", (False, True))
def test_fix20_token_rollback_orders_newer_generation_after_failed_claim(
        monkeypatch, error_type, units, committed_prefix):
    engine, entry, holder, peer, bucket = _fix20_pool(units)
    # Real retired generations put the queue across grant:9 / grant:10.
    for _ in range(7):
        pad = PrefetchPoolProgram(holder, 1, units).start(engine, "pad", entry)
        end = ReleasePoolProgram(holder).start(engine, "pad:release", pad)
        engine.advance(until=(end,), first=False)
    copy = PrefetchPoolProgram(holder, 1536, units, lease_token="T")
    release = ReleasePoolProgram(holder, lease_token="T")
    first = copy.start(engine, "r0:i0:copy", entry)
    gate = engine.submit("consumer", after=(first,), latency=10)
    committed = release.start(engine, "old:release", gate) if committed_prefix else None
    target = copy.start(engine, "r0:i1:copy", entry) if committed_prefix else first
    grant = _fix20_grant(bucket, target)
    submit = engine.submit
    sentinel = error_type("free with newer token generation")
    newer_handles = []

    def fail(name, *args, **kwargs):
        if name == "target:release:free":
            assert kwargs["free"] == (grant.allocation,)
            newer_handles.append(copy.start(engine, "r0:i2:copy", entry))
            raise sentinel
        return submit(name, *args, **kwargs)

    with monkeypatch.context() as injected:
        injected.setattr(engine, "submit", fail)
        with pytest.raises(error_type) as caught:
            release.start(engine, "target:release", gate)
    assert caught.value is sentinel and not grant.release_scheduled
    newer = _fix20_grant(bucket, newer_handles[0])
    assert bucket.token_live["T"] == [grant.grant_id, newer.grant_id]
    # Rollback is idempotent and cannot move either generation or the peer.
    holder.rollback_release(engine, grant)
    assert bucket.token_live["T"] == [grant.grant_id, newer.grant_id]
    assert bucket.token_live["peer"] == [_fix20_grant(bucket, peer).grant_id]
    if committed_prefix:
        older = _fix20_grant(bucket, first)
        assert older.release_scheduled and not older.retired
        assert older.grant_id not in bucket.token_live["T"]
    retry = release.start(engine, "retry:release", gate)
    assert retry.name == "retry:release:free" and grant.release_scheduled
    assert not newer.release_scheduled and bucket.token_live["T"] == [newer.grant_id]
    last = release.start(engine, "last:release", gate)
    assert newer.release_scheduled and "T" not in bucket.token_live
    engine.advance(until=(target, newer_handles[0]), first=False)
    assert not grant.retired and not newer.retired and not gate.done
    assert engine.summary()["live_allocated"] == {"l1": 768.0 + 1536 * (2 + committed_prefix)}
    engine.advance(until=(retry, last, *((committed,) if committed else ())), first=False)
    assert grant.retired and newer.retired
    assert retry.time == last.time == gate.time
    assert engine.summary()["live_allocated"] == {"l1": 768.0}
    again = copy.start(engine, "r0:i0:copy", entry)
    reused = _fix20_grant(bucket, again)
    holder.rollback_release(engine, grant)  # Retired capabilities stay retired.
    assert bucket.token_live["T"] == [reused.grant_id]
    finish = release.start(engine, "again:release", again)
    peer_free = ReleasePoolProgram(holder).start(engine, "peer:release", peer)
    engine.advance(until=(finish, peer_free), first=False)
    assert not bucket.active_by_key and not bucket.token_live
    assert engine.summary()["live_allocated"] == {"l1": 0.0}


def test_fix20_rollback_middle_generation_and_foreign_ownership():
    engine, entry, holder, peer, bucket = _fix20_pool(1)
    claims = []
    for index in range(12):
        copied = PrefetchPoolProgram(holder, 128, 1, lease_token="T").start(
            engine, f"r0:i{index}:copy", entry)
        claims.append(holder.claim_release(engine, handle=copied))
    for index in (11, 0, 5, 9, 2, 10, 4, 8, 1, 7, 3, 6):
        holder.rollback_release(engine, claims[index])
    expected = [g.grant_id for g in claims]
    assert bucket.token_live["T"] == expected
    middle = claims[5]
    holder.rollback_release(engine, middle)
    assert bucket.token_live["T"] == expected  # No duplicate insertion.
    foreign = Engine([])
    holder.rollback_release(foreign, middle)
    assert not holder._bucket(foreign).token_live and bucket.token_live["T"] == expected
    bucket.active_by_key.pop(middle.key)
    holder._forget_token(bucket, middle)
    holder.rollback_release(engine, middle)
    assert bucket.token_live["T"] == [g for g in expected if g != middle.grant_id]
    bucket.active_by_key[middle.key] = middle
    holder.rollback_release(engine, middle)
    assert bucket.token_live["T"] == expected
    assert bucket.token_live["peer"] == [_fix20_grant(bucket, peer).grant_id]
