"""C11 ``_engine_accel`` bridge hand-back, snapshot order and causal finalizers.

[Role] Guard the in-memory runtime bridge (``accel_bridge.py``,
  ``accel_sync.py``): backend selection/load-failure fallback, GC context
  finalizers, aborted publication preserving primary exceptions, hand-back
  reconciliation of dependencies/residues/callbacks, completion and service
  snapshot trace order, native wire ``F_CALLBACK`` admission holds, Engine
  sentinel-pool attachment phases, and causal finalizer / ``_apply`` declines.
[Invariants] Every test compares native ``c`` and ``auto`` backends field-for-
  field against the canonical Python ``Engine`` using only public APIs and
  deterministic CPython ``weakref.finalize`` triggers.
[Timeouts & Locks] No threads, sleeps or subprocesses; uses the session-cached
  compiled library only.
[Side-Effect Policy] In-memory only; writes nothing to disk.
"""
from __future__ import annotations

import gc
import weakref

import pytest

from _support.accel import (
    BACKENDS, FRONTIER_WORK, NATIVE, Token, arm_observer, close,
    frontier_roots, make_counter_observer, native_library, uniform_engine,
)
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.runtime import (
    accel_bridge, accel_build, accel_sync, accel_tables as tables,
)
from mindie_sim.prediction.runtime.accel_tables import MAX_OWN
from mindie_sim.prediction.runtime.engine import (
    _EMPTY_FROZENSET as ENGINE_EMPTY_FROZENSET,
)

pytestmark = pytest.mark.tier0


# -- backend selection, load-failure fallback & context finalizers ----------
def test_backend_selection_summary_and_load_failure_fallback(monkeypatch):
    tiny = Engine([Resource("r", 1000.0)], backend="auto")
    for i in range(8):
        tiny.submit(f"t{i}", {"r": 1.0})
    tiny.advance()
    assert tiny.accel_status()["native_calls"] == 0
    assert tiny.accel_status()["accel_backend"] in {"python", "python_fallback"}

    burst = Engine([Resource("r", 1000.0)], backend="auto")
    burst.advance(until=tuple(burst.submit(f"t{i}", {"r": 1.0}) for i in range(64)))
    summary = burst.summary()
    assert summary["accel_backend"] == "c_native" and summary["native_steps"] > 0
    assert summary["native_calls"] > 0 and summary["fallback_reason"] is None
    close(burst)

    for backend in BACKENDS:
        eng = Engine([Resource("r", 1.0)], backend=backend)
        eng.submit("sleep", {}, latency=10.0)
        eng.advance(horizon=1.0)
        eng.submit("work", {"r": 1.0}, on_complete=lambda _e: None)
        eng.advance()
        assert eng.summary()["fallback_reason"] == eng.accel_status()["fallback_reason"]
        assert eng.summary()["accel_backend"] == eng.accel_status()["accel_backend"]
        if backend == "c":
            assert eng.summary()["fallback_reason"] == "mixed_frontier"
        close(eng)

    with monkeypatch.context() as mp:
        mp.setattr(accel_build, "discover_compiler",
                   lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("touched")))
        py = Engine([Resource("r", 10.0)], backend="python")
        assert py.accel_status()["accel_backend"] == "python"
        assert py.accel_status()["fallback_reason"] == "requested_python"

    with monkeypatch.context() as mp:
        mp.setattr(accel_bridge, "load_library", lambda st: (_ for _ in ()).throw(
            accel_bridge.AccelUnavailable("library_load_failed:injected", st)))
        with pytest.raises(accel_bridge.AccelUnavailable):
            Engine([Resource("r", 10.0)], backend="c")
        auto = Engine([Resource("r", 10.0)], backend="auto")
        events = [auto.submit(f"t{i}", {"r": 1.0}) for i in range(64)]
        auto.advance(until=tuple(events))
        status = auto.accel_status()
        assert status["native_steps"] == 0 and all(e.done for e in events)
        assert "injected" in (status["backend_reason"] or "")


def test_context_finalizers_free_plain_and_cyclic_engines_exactly_once(monkeypatch):
    library = native_library()
    calls, original = [], library.msim_ctx_free
    monkeypatch.setattr(library, "msim_ctx_free",
                        lambda ctx: (calls.append(ctx), original(ctx))[1])
    gc.collect()
    baseline = len(calls)

    for backend in ("python", "auto"):
        refs = []
        for i in range(3):
            eng = Engine([Resource("r", 10.0)], backend=backend)
            eng.submit(f"pending{i}", {"r": 1.0}, metadata={"engine": eng})
            refs.append(weakref.ref(eng))
            del eng
        gc.collect()
        assert all(ref() is None for ref in refs), backend

    refs = []
    for i in range(3):
        eng = Engine([Resource("r", 10.0)], backend="c")
        eng.submit(f"pending{i}", {"r": 1.0}, metadata={"engine": eng})
        refs.append(weakref.ref(eng))
        del eng
    gc.collect()
    assert all(ref() is None for ref in refs) and len(calls) - baseline == 3

    eng = Engine([Resource("r", 10.0)], backend="c")
    eng.submit("pending", {"r": 1.0}, metadata={"engine": eng})
    eng._accel.close()
    assert len(calls) - baseline == 4
    del eng
    gc.collect()
    assert len(calls) - baseline == 4


def test_aborted_publication_preserves_primary_exception(monkeypatch):
    eng = Engine([Resource("r", 128.0)], backend="c", trace_limit=0)
    first = eng.submit("root0", {"r": 1.0}, caps={"r": 1.0})
    second = eng.submit("root1", {"r": 1.0}, caps={"r": 1.0})
    roots = [eng.submit(f"root{i}", {"r": 1.0}, caps={"r": 1.0}) for i in range(2, 10)]
    tail = eng.submit("tail", {"r": 2.0}, caps={"r": 1.0})
    original = eng._complete_native

    def boom(task, moment):
        if task.event.name == "root1":
            eng.on_complete(roots[-1], lambda _e: None)
            raise RuntimeError("primary-callback-boom")
        return original(task, moment)

    monkeypatch.setattr(eng, "_complete_native", boom)
    with pytest.raises(RuntimeError, match="primary-callback-boom") as caught:
        eng.advance()
    assert caught.value.__context__ is None
    assert eng._accel.aborted_publication is True and eng._accel.pending_handback is True
    assert eng.completed_count == 1 and len(eng._native_slots) == 9
    assert first.time == 1.0 and second.time is None and tail.time is None and roots[-1].time is None
    close(eng)


# -- hand-back dependency, residue and callback reconciliation --------------
def _handback_case(backend, mode):
    if mode in ("chain", "diamond", "adopted"):
        eng = Engine([Resource("r", 1.0)], backend=backend)
        a = eng.submit("A", {"r": 1.0})
        b = eng.submit("B", {"r": 1.0}, after=(a,))
        if mode == "diamond":
            c = eng.submit("C", {"r": 1.0}, after=(a,))
            evs, stop = [a, b, c, eng.submit("D", {"r": 1.0}, after=(b, c))], (a,)
        else:
            c = eng.submit("C", {"r": 1.0}, after=(b,))
            rest = [eng.submit(f"x{i}", {"r": 1.0}) for i in range(61)] if mode == "adopted" else []
            evs, stop = [a, b, c, *rest], (a if mode == "adopted" else b,)
        eng.advance(until=stop)
        z = eng.submit("Z", {"r": 1.0}, on_complete=lambda _e: None)
        eng.advance()
        out = (eng.now, [e.time for e in (*evs, z)], eng.accel_status()["native_steps"])
    elif mode == "delayed":
        eng = Engine([Resource("r", 64.0)], backend=backend)
        for i in range(62):
            eng.submit(f"w{i}", {})
        root = eng.submit("A", {"r": 1.0}, release=2.0)
        fol = eng.submit("B", {"r": 1.0}, after=(root,))
        eng.advance(horizon=1.0)
        d = eng.submit("D", {"r": 1.0}, on_complete=lambda _e: None)
        eng.advance()
        out = (eng.now, root.started, root.time, fol.time, d.time, len(eng._releases))
    elif mode == "latency":
        eng = Engine([Resource("r", 1.0)], backend=backend)
        for i in range(63):
            eng.submit(f"w{i}", {})
        sleep = eng.submit("sleep", {"r": 1.0}, latency=10.0, coupled=True)
        eng.advance(horizon=2.0)
        d = eng.submit("D", {"r": 1.0}, on_complete=lambda _e: None)
        eng.advance()
        out = (eng.now, sleep.time, d.time, dict(eng.serviced), eng._tasks == {})
    elif mode == "partial":
        eng = Engine([Resource("r1", 100.0), Resource("r2", 1.0)], backend=backend)
        for i in range(63):
            eng.submit(f"w{i}", {})
        mixed = eng.submit("M", {"r1": 1.0, "r2": 10.0})
        eng.advance(horizon=0.5)
        d = eng.submit("D", {"r1": 1.0}, on_complete=lambda _e: None)
        eng.advance(horizon=0.5001)
        task = eng._tasks[mixed.sequence]
        assert all(v > 0.0 for v in task.remaining.values())
        snap = (dict(task.remaining), dict(task.service))
        eng.advance()
        out = (snap, eng.now, mixed.time, d.time, dict(eng.serviced))
    else:
        count = 4097 if mode == "after_drain" else 64
        eng, evs = uniform_engine(backend, count, rate=float(count) if mode == "after_drain" else 1.0)
        if mode == "after_drain":
            eng.advance(horizon=0.0)
            target = evs[4096]
        else:
            target = eng.submit("a", {"r": 1.0})
            eng.submit("b", {"r": 2.0})
            eng.advance(horizon=0.5)
        seen, added = [], []
        eng.on_complete(target, lambda e: (seen.append((e.name, e.time, eng.now)),
                                           added.append(eng.submit("c", {"r": 1.0}))))
        eng.advance()
        out = (eng.now, seen, added[0].started, added[0].time, eng.completed_count)
    close(eng)
    return out


@pytest.mark.parametrize("mode", ("chain", "diamond", "adopted", "delayed",
                                  "latency", "partial", "late_cb", "after_drain"))
def test_handback_reconciles_dependencies_residues_and_callbacks(mode):
    results = {backend: _handback_case(backend, mode) for backend in BACKENDS}
    for backend in NATIVE:
        if mode in ("chain", "diamond", "adopted"):
            assert results[backend][:2] == results["python"][:2], (mode, backend)
            assert backend != "c" or results[backend][2] > 0
        else:
            assert results[backend] == results["python"], (mode, backend)


# -- snapshot trace order, early child & native wire F_CALLBACK hold --------
@pytest.mark.parametrize(("count", "nres", "latency"), (
    (4097, 1, False), (1025, 32, False), (4097, 1, True),
))
def test_snapshot_continuation_preserves_early_child_trace_order(count, nres, latency):
    names = [f"r{i}" for i in range(nres)]
    work = {} if latency else dict.fromkeys(names, 1.0)
    kw = {"latency": 1.0} if latency else {}
    expected = [f"root{i}" for i in range(count)] + ["child"]
    for limit in (1, 3, count + 2):
        for backend in BACKENDS:
            eng = Engine([Resource(n, float(count)) for n in names],
                         backend=backend, trace_limit=limit)
            first = eng.submit("root0", work, **kw)
            child = eng.submit("child", {}, after=(first,))
            for i in range(1, count):
                eng.submit(f"root{i}", work, **kw)
            eng.advance(horizon=1.0, max_steps=1)
            assert [row["task"] for row in eng.trace] == expected[-limit:], (count, nres, limit, backend)
            assert (child.started, child.time, eng.completed_count) == (1.0, 1.0, count + 1)
            assert backend == "python" or eng.accel_status()["native_steps"] > 0
            close(eng)


@pytest.mark.parametrize(("count", "nres", "trigger"), ((4097, 1, 4095), (1025, 32, 1023)))
def test_native_bridge_holds_admission_until_flagged_snapshot_publishes(
        monkeypatch, count, nres, trigger):
    names = [f"r{i}" for i in range(nres)]
    work = dict.fromkeys(names, 1.0)
    orig_flags = accel_sync.task_flags
    flag_name = f"root{trigger}"
    monkeypatch.setattr(
        accel_sync, "task_flags",
        lambda t, e=None: orig_flags(t, e) | tables.F_CALLBACK if t.event.name == flag_name
        else orig_flags(t, e),
    )
    out = {}
    for backend in BACKENDS:
        eng = Engine([Resource(n, float(count)) for n in names], backend=backend, trace_limit=0)
        roots = [eng.submit(f"root{i}", work) for i in range(count)]
        child = eng.submit("child", work, after=(roots[trigger],))
        seen, orig_comp = [], eng._complete_native

        def traced(task, moment, _orig=orig_comp, _seen=seen, _child=child):
            if task.event.name in (flag_name, f"root{trigger + 1}"):
                _seen.append((task.event.name, _child.started, moment))
            return _orig(task, moment)

        eng._complete_native = traced
        eng.advance(horizon=1.0)
        assert (child.started, child.time) == (1.0, None), (backend, count, nres)
        eng.advance()
        out[backend] = (seen, (child.started, child.time), eng.now, eng.completed_count,
                        eng.accel_status()["native_steps"], len(eng._native_slots))
        close(eng)
    for backend in NATIVE:
        assert out[backend][1:4] == out["python"][1:4] and out[backend][4] > 0 and out[backend][5] == 0
        assert out[backend][0] == [(flag_name, None, 1.0), (f"root{trigger + 1}", None, 1.0)]


# -- sentinel attachment phases & causal finalizers -------------------------
@pytest.mark.parametrize("snapshot", (False, True), ids=("module", "snapshot"))
@pytest.mark.parametrize("phase,horizon,steps", (
    ("before_submit", None, 0), ("after_submit_before_advance", None, 3),
    ("after_zero_horizon", 0.0, 3), ("after_partial_advance", 0.5, 4),
))
def test_sentinel_pool_attachment_phases_keep_python_parity(
        monkeypatch, snapshot, phase, horizon, steps):
    submit, run, control = tables.Arena.submit, tables.Arena.run, None
    for backend in BACKENDS:
        eng = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
        pool = eng._memory_pools_frozen if snapshot else ENGINE_EMPTY_FROZENSET
        flags, calls, seen, finalizer = [], [], [], None

        def packed(arena, rows, deps=()):
            flags.extend(row[1] for row in rows)
            return submit(arena, rows, deps)

        def ran(arena, *args):
            answer = run(arena, *args)
            calls.append((answer[1], arena.last_held))
            return answer

        monkeypatch.setattr(tables.Arena, "submit", packed)
        monkeypatch.setattr(tables.Arena, "run", ran)
        try:
            if phase == "before_submit":
                finalizer = weakref.finalize(pool, seen.append, "released")
            kw = {"metadata": {"inert": 1}} if snapshot else {}
            roots = [eng.submit(f"root{i}", {"r": w}, caps={"r": 1.0}, **kw)
                     for i, w in enumerate((2.0,) + (3.0,) * 62 + (1.0,))]
            child = eng.submit("child", {"r": 1.0}, caps={"r": 1.0}, after=(roots[-1],), **kw)
            if horizon is not None:
                eng.advance(horizon=horizon)
            if finalizer is None:
                finalizer = weakref.finalize(pool, seen.append, "released")
            del pool
            eng.advance()
            semantic = (eng.now, eng.serviced["r"], eng.completed_count, (child.started, child.time))
            control = semantic if backend == "python" else control
            assert semantic == control and finalizer.alive and not seen and not eng._native_slots
            native = backend != "python" and phase != "before_submit"
            cb = native and phase == "after_submit_before_advance"
            assert flags == ([tables.F_CALLBACK if cb else 0] * 65 if native else [])
            assert sum(h for _, h in calls) == (3 if cb else 0)
            assert eng.accel_status()["native_steps"] == (steps if native else 0)
        finally:
            if finalizer is not None:
                finalizer.detach()
            close(eng)


def _finalizer_case(backend, kind):
    if kind in ("cb_work", "counters", "wide_prio", "cap_decline"):
        count = 4097 if kind in ("cb_work", "counters", "cap_decline") else 64
        eng = Engine([Resource("r", float(count))], backend=backend, trace_limit=0)
        tok = Token()
        roots = [eng.submit("root0", {"r": 1.0}, metadata={"token": tok})] + [
            eng.submit(f"root{i}", {"r": 1.0}) for i in range(1, count)]
        reg, obs, extra = [], [], []

        def follow(ev):
            obs.append((ev.time, eng.now, eng.busy_ms["r"], eng.serviced["r"]))
            if kind == "cb_work":
                e1 = eng.submit("e1", {"r": 1.0})
                extra.extend([e1, eng.submit("e2", {"r": 1.0}, after=(e1,))])
            elif kind == "counters":
                t = eng.submit("e", {"r": eng.serviced["r"]}, on_complete=lambda _e: None)
                extra.append(t)
                eng.on_complete(eng.start_boundary(t),
                                lambda o: obs.append((o.time, o.started, eng.serviced["r"])))
            elif kind == "wide_prio":
                extra.append(eng.submit("e", {"r": 1.0}, priority=2 ** 63))
            else:
                waits = [eng.future(f"w{i}") for i in range(MAX_OWN + 1)]
                extra.append(eng.submit("e", {"r": 1.0}, after=tuple(waits)))
                for w in waits:
                    eng.resolve(w)

        fin = arm_observer(eng, tok, roots[-1 if kind in ("cb_work", "counters") else 1], follow, reg)
        del tok
        eng.advance()
        assert not fin.alive
        out = (reg, obs, [(e.started, e.time) for e in extra], eng.now,
               eng.serviced["r"], eng.completed_count)
    elif kind in ("trace_1", "trace_2", "mutable"):
        limit = 0 if kind == "mutable" else int(kind.split("_")[1])
        eng = Engine([Resource("r", 64.0)], backend=backend, trace_limit=limit)
        tok, box = Token(), []
        roots = frontier_roots(eng, metadata={"p": box} if kind == "mutable" else {"token": tok})
        if kind == "mutable":
            eng.advance(horizon=0.0)
            box.append(tok)
        register, reg, obs, extra = make_counter_observer(eng, roots, 1)
        fin = weakref.finalize(tok, register)
        del tok, box
        eng.advance()
        out = (reg, obs, [(e.started, e.time) for e in extra], eng.now,
               [r["task"] for r in eng.trace], not fin.alive)
    else:
        eng = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
        tok, seen = Token(), []
        if kind == "descendant":
            for i, a in enumerate(FRONTIER_WORK[1:]):
                eng.submit(f"r{i}", {"r": a}, caps={"r": 1.0})
            r0 = eng.submit("root63", {"r": 1.0}, caps={"r": 1.0}, metadata={"token": tok})
            ch = eng.submit("child", {"r": 1.0}, caps={"r": 1.0}, after=(r0,))
            weakref.finalize(tok, lambda: (seen.append((eng.now, r0.done, ch.started)),
                                           eng.submit("e", {"r": 1.0 if ch.started is None else 2.0})))
            del tok
            eng.advance()
        else:
            r0 = eng.submit("root0", {}, metadata={"token": tok})
            ch = eng.submit("child", {"r": 1.0}, after=(r0,))
            weakref.finalize(tok, lambda: seen.append((eng.now, r0.done, ch.started)))
            del tok
            eng.advance(horizon=0.0 if kind == "zero_horizon" else float("inf"))
        out = (seen, eng.now, eng.serviced["r"], eng.completed_count)
    close(eng)
    return out


@pytest.mark.parametrize("kind", (
    "cb_work", "counters", "wide_prio", "cap_decline", "trace_1", "trace_2",
    "mutable", "descendant", "zero_horizon", "zero_chain",
))
def test_causal_finalizers_and_apply_declines_match_python(kind):
    results = {backend: _finalizer_case(backend, kind) for backend in BACKENDS}
    for backend in NATIVE:
        assert results[backend] == results["python"], (kind, backend)


def test_periodic_template_blocks_rearm_during_callbacks_then_resumes_past_int32_sequence():
    from mindie_sim.prediction.program import PipelinePhase, PipelineTemplate, Stage
    from mindie_sim.prediction.runtime.accel_sync import REARM_MIN_TASKS, can_rearm
    from mindie_sim.prediction.scheduling import start_pipeline_template

    outcomes = {}
    for backend in ("python", "auto"):
        eng = Engine([Resource("r", 100.0)], backend=backend)
        template = PipelineTemplate(
            owner=0,
            slots=2,
            phases=(PipelinePhase("main", (Stage("work", {"r": 100.0}, previous=("work",)),), (1 << 31) + 100),),
        )
        done = start_pipeline_template(eng, template, "long", eng.submit("ready"))
        if backend == "auto" and eng._accel is not None:
            eng.advance(horizon=0.0)
            assert eng._inline_blocked
            assert not can_rearm(eng._accel, eng)
        eng.advance(until=(done,), max_steps=100000)
        tails = tuple(eng.submit(f"tail{idx}", {"r": 100.0}) for idx in range(REARM_MIN_TASKS))
        assert tails[0].sequence > (1 << 31)
        eng.advance(until=tails, max_steps=REARM_MIN_TASKS + 10)
        outcomes[backend] = (eng.now, eng.serviced["r"], [t.time for t in tails])
        close(eng)
    assert outcomes["auto"] == outcomes["python"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_hbm_bus_directional_aliasing_preprepared_and_native_hook(backend):
    from mindie_sim.prediction.runtime.memory_physics import HbmBusSpec

    # Equal-rate alpha=0: shipping read and independent catalog write overlap in 8 ms.
    bus_eq = HbmBusSpec(0, 1024.0, 1024.0, 0.0, row_r="memory@r0:s0:r", row_w="memory@r0:s0:w")
    eng = Engine([Resource("memory@r0:s0", 1024.0), Resource("mte2_aic@r0:s0", 8192.0)], backend=backend)
    eng.add_hbm_bus(bus_eq)
    ev_r = eng.submit("ship_read", {"memory@r0:s0": 8192.0, "mte2_aic@r0:s0": 8192.0})
    ev_w = eng.submit("cat_write", {"memory@r0:s0:w": 8.0})
    eng.advance(until=(ev_r, ev_w), first=False)
    assert eng.now == pytest.approx(8.0)
    close(eng)

    # Pre-prepared stage refreshes against later bus registration (16 ms).
    eng2 = Engine([Resource("memory@r0:s0", 1024.0)], backend=backend)
    prep = eng2.prepare_stage("ship_read", {"memory@r0:s0": 8192.0})
    eng2.add_hbm_bus(bus_eq)
    e1 = eng2.submit_prepared(prep, "ship_read")
    e2 = eng2.submit("cat_read", {"memory@r0:s0:r": 8.0})
    eng2.advance(until=(e1, e2), first=False)
    assert eng2.now == pytest.approx(16.0)
    close(eng2)

    # Pre-existing directional rows register aliases immediately (16 ms).
    eng3 = Engine([
        Resource("memory@r0:s0", 1024.0),
        Resource("memory@r0:s0:r", 1.0, unit="row_fraction", policy="fair", units=1),
        Resource("memory@r0:s0:w", 1.0, unit="row_fraction", policy="fair", units=1),
    ], backend=backend)
    e1 = eng3.submit("ship_read", {"memory@r0:s0": 8192.0})
    e2 = eng3.submit("cat_read", {"memory@r0:s0:r": 8.0})
    eng3.advance(until=(e1, e2), first=False)
    assert eng3.now == pytest.approx(16.0)
    close(eng3)

    # Registering bus from a native_hook=True callback defers native-slot retrofit cleanly.
    eng4 = Engine([Resource("memory@r0:s0", 1024.0)], backend=backend)
    ship = eng4.submit("ship_read", {"memory@r0:s0": 10240.0})
    eng4.submit("hook", latency=3.0, on_complete=lambda _ev: eng4.add_hbm_bus(bus_eq), native_hook=True)
    eng4.advance(until=(ship,))
    assert ship.time == pytest.approx(10.0)
    close(eng4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_tensor_view_alias_waits_for_backing_dirty_writeback(backend):
    from mindie_sim.prediction.runtime.memory_physics import (
        AddressRun, AddressRunPlan, AdmissionMemoryView, HbmBusSpec,
        MemoryAccessPattern, ReuseKey, VisibilityContract, bind_memory_access,
    )

    size = 64
    bus = HbmBusSpec(0, 64.0, 16.0, 0.0)
    view = AdmissionMemoryView({0: size}, granule_bytes=size)
    eng = Engine([Resource("mte3", 32.0, units=1), Resource("mte2_aiv", 64.0, units=1),
                  Resource("l2", 64.0, units=1)], backend=backend, trace_limit=64)
    eng.add_hbm_bus(bus)
    key = ReuseKey(0, "buf_X", 1, "s0")
    runs = AddressRunPlan((AddressRun(0, 1, size, size),))
    p_wr = MemoryAccessPattern("X", 1, 0, 0, "write", "mte3", size, runs, key,
                               visibility=VisibilityContract(scope="die", private_workspace=True))
    p_rd = MemoryAccessPattern("X", 1, 0, 0, "read", "mte2_aiv", size, runs, key,
                               visibility=VisibilityContract(scope="die", private_workspace=True))
    b_wr = bind_memory_access(p_wr, view, hbm_bus=bus, copy_hit_bytes_per_ms=32.0, copy_miss_bytes_per_ms=32.0,
                              l2_bytes_per_ms=64.0, pipe_row_id="mte3", l2_row_id="l2", commit=False)
    b_rd = bind_memory_access(p_rd, view, hbm_bus=bus, copy_hit_bytes_per_ms=64.0, copy_miss_bytes_per_ms=64.0,
                              l2_bytes_per_ms=64.0, pipe_row_id="mte2_aiv", l2_row_id="l2", commit=False)
    ev_w = eng.submit_bound_access("wr", b_wr, release=0.0, memory_state=view, pattern=p_wr)
    eng.advance(until=(ev_w,))
    eng.advance(horizon=2.5)
    view.grant_reservation(0, "full", size)
    r1 = eng.submit_bound_access("r1", b_rd, release=3.0, memory_state=view, pattern=p_rd)
    r2 = eng.submit_bound_access("r2", b_rd, release=4.1, memory_state=view, pattern=p_rd)
    eng.advance(until=(r1, r2), first=False)
    wb_end = next(r["end_ms"] for r in eng.trace if "dirty_writeback" in r["task"])
    assert r1.time >= wb_end and r2.time >= wb_end and eng.delivered[bus.ctr_w] == float(size)
    close(eng)
