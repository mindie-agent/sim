"""Python reference == C11 ``_engine_accel`` numerics, eligibility and ABI.

[Role] Guard the engine-arena/2 accelerator as a drop-in for the canonical
  Python ``Engine``: seeded random programs sized from SKU geometry agree to
  <=8 ULP and take the native path; standard ``KernelProgram``/``Stage`` graphs
  run natively; boundary programs (output-buffer continuations, placement pools,
  delayed releases, start boundaries, arithmetic prefixes) are bit-identical;
  the exact-builtin eligibility rule declines every weakrefable/mutable payload
  and ``"occurrence"``; coupled placements must be proportional; ``MAX_R/U``
  ceilings and raw ABI preflights reject malformed input before mutation.
[Invariants] The Python backend is the oracle in every comparison.  Two known
  coupled parity defects are pinned as strict xfails: accumulation drift >8 ULP
  (``_engine_accel_rates.h:471``) and disjoint-coupled + overlapping-uncoupled
  solver dispatch mismatch (``arbiter.py:198``).
[Timeouts & Locks] No sleeps; one bounded thread pool; the compiled library
  comes from the session build memo.
[Side-Effect Policy] In-memory only; no subprocesses, no file writes.
"""
from __future__ import annotations

import concurrent.futures
import ctypes
import dataclasses
import enum
import math
import types
import typing
import weakref

import pytest

from _support.accel import (
    BACKENDS, BOUNDARY_CASES, FRONTIER_WORK, NATIVE, Token, arena_row, close,
    max_ulps, native_library, open_arena, random_program, run_program,
    sku_resources, ulps, uniform_engine,
)
from _support.axes import REPRESENTATIVE_SKUS, SKUS
from mindie_sim.prediction.engine import Engine, ExecutionError, Resource
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.prediction.runtime import accel_build, accel_tables as tables
from mindie_sim.prediction.runtime.accel_tables import ABI_VERSION, KernelFault, PROBE_GOLDEN
from mindie_sim.prediction.runtime.engine import _EMPTY_FROZENSET as ENGINE_EMPTY_FROZENSET

pytestmark = pytest.mark.tier0
ULP_BOUND = 8
INF = float("inf")


class _ParityGapError(AssertionError):
    """Raised only after native execution completes cleanly yet exceeds ULP target."""


# -- seeded random programs & standard graphs -------------------------------
@pytest.mark.parametrize("sku", SKUS)
@pytest.mark.parametrize("seed", (0, 1))
def test_uncoupled_random_programs_match_python_on_the_native_path(sku, seed):
    resources, tasks = random_program(seed, sku, kinds=(0, 1, 3, 4, 5))
    python, native = (run_program(resources, tasks, b) for b in ("python", "c"))
    assert native["steps"] > 0 and native["reason"] is None
    assert native["completed"] == python["completed"] == len(tasks)
    assert max_ulps(python, native) <= ULP_BOUND


@pytest.mark.parametrize("sku", SKUS)
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_coupled_random_programs_match_python_on_the_native_path(sku, seed):
    resources, tasks = random_program(seed, sku, size=12, kinds=(0, 1, 2, 3, 5))
    assert any(spec.get("coupled") for spec in tasks)
    python, native = (run_program(resources, tasks, b) for b in ("python", "c"))
    assert native["steps"] > 0 and native["reason"] is None
    assert max_ulps(python, native) <= ULP_BOUND


@pytest.mark.parametrize(("build", "target_ulps"), (
    pytest.param(
        lambda: ([Resource("a", 26.289056632197735), Resource("b", 36.495015663833634)], [
            {"work": {"a": 7.05924503470419, "b": 8.494354162581535}, "coupled": True},
            {"work": {"a": 6.694301119164283, "b": 5.5353786183562095}, "coupled": True},
            {"work": {"a": 8.483307144965345, "b": 7.87160639063931}, "coupled": True},
        ]), 0, id="coupled-3task-bit-exact",
        marks=pytest.mark.xfail(strict=True, raises=_ParityGapError, reason=(
            "coupled 3-task 2-ULP order drift: mindie_sim/prediction/runtime/_engine_accel_rates.h:471")),
    ),
    pytest.param(
        lambda: random_program(2, REPRESENTATIVE_SKUS["a5"], size=96), ULP_BOUND, id="coupled-96task-8ulp",
        marks=pytest.mark.xfail(strict=True, raises=_ParityGapError, reason=(
            "coupled 96-task >8-ULP accumulation drift: mindie_sim/prediction/runtime/_engine_accel_rates.h:471")),
    ),
    pytest.param(
        lambda: ([Resource("cube", 245.76, units=20), Resource("vector", 1.92, units=40)], [
            {"work": {"cube": 1.0154159211083948}, "placement": {"cube": (10, 15, 17)}},
            {"work": {"cube": 3.6808735333841436},
             "placement": {"cube": (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19)}},
            {"work": {"cube": 500.0, "vector": 100.0}, "coupled": True,
             "placement": {"cube": (3, 4), "vector": (6, 7, 8, 9)}},
        ]), ULP_BOUND, id="disjoint-coupled-solver",
        marks=pytest.mark.xfail(strict=True, raises=_ParityGapError, reason=(
            "disjoint coupled + overlapping uncoupled solver mismatch: mindie_sim/prediction/runtime/arbiter.py:198")),
    ),
))
def test_coupled_parity_gap_witnesses_run_natively_and_track_ulp_targets(build, target_ulps):
    resources, tasks = build()
    python, native = (run_program(resources, tasks, b) for b in ("python", "c"))
    assert native["steps"] > 0 and native["reason"] is None
    assert native["completed"] == python["completed"] == len(tasks)
    observed = max_ulps(python, native)
    if observed > target_ulps:
        raise _ParityGapError(f"observed {observed} ULP > target {target_ulps}")


@pytest.mark.parametrize("sku", SKUS)
def test_standard_stage_graph_takes_native_path_without_fallback(sku):
    resources = sku_resources(sku)
    cube, vector = resources[0], resources[1]
    width = cube.units // 2
    aiv = width * vector.units // cube.units
    program = KernelProgram((
        Stage("load", {"mte2": resources[2].rate}),
        Stage("fused", {"cube": cube.rate, "vector": vector.rate}, after=("load",),
              coupled=True, placement={"cube": tuple(range(width)), "vector": tuple(range(aiv))}),
        Stage("store", {"hbm": resources[4].rate}, after=("fused",)),
    ))
    results = {}
    for backend in BACKENDS:
        engine = Engine(resources, backend=backend)
        ready = engine.submit("ready")
        done = [program.start(engine, f"g{i}", ready) for i in range(24)]
        engine.advance(until=tuple(done))
        status = engine.accel_status()
        results[backend] = (engine.now, status["native_steps"], status["fallback_reason"])
        close(engine)
    for backend in NATIVE:
        assert results[backend][1] > 0 and results[backend][2] is None
        assert results[backend][0] == pytest.approx(results["python"][0], rel=1e-13)


def test_build_flags_forbid_value_changing_float_optimisation():
    flags = set(accel_build.compiler_flags())
    assert {"-fno-fast-math", "-ffp-contract=off"} <= flags
    assert not {"-ffast-math", "-Ofast", "-march=native"} & flags


# -- bit-identical boundary programs ----------------------------------------
@pytest.mark.parametrize(("builder", "auto_native"), [pytest.param(fn, ok, id=k) for k, fn, ok in BOUNDARY_CASES])
def test_boundary_programs_are_bit_identical_to_python(builder, auto_native):
    observed, steps, reasons = {}, {}, {}
    for backend in BACKENDS:
        engine, observed[backend] = builder(backend)
        status = engine.accel_status()
        steps[backend], reasons[backend] = status["native_steps"], status["fallback_reason"]
        close(engine)
    assert steps["c"] > 0 and reasons["c"] is None
    if auto_native:
        assert steps["auto"] > 0 and reasons["auto"] is None
    for backend in NATIVE:
        assert observed[backend] == observed["python"], backend


def test_start_boundary_observation_and_causality_match_python():
    for mode in ("pure", "until", "dependency", "callback"):
        rows = {}
        for backend in BACKENDS:
            engine = Engine([Resource("r", 1.0)], backend=backend)
            for i in range(63):
                engine.submit(f"w{i}", {})
            work = engine.submit("work", {"r": 1.0})
            marker = engine.start_boundary(work)
            if mode == "pure":
                engine.advance()
                rows[backend] = (marker.time, marker.started, work.time, engine.now)
                assert backend == "python" or engine.accel_status()["native_steps"] > 0
            elif mode == "until":
                engine.advance(until=(marker,))
                rows[backend] = (marker.time, work.started, work.time, engine.now)
            elif mode == "dependency":
                follower = engine.submit("f", {"r": 1.0}, after=(marker,))
                engine.advance()
                rows[backend] = (marker.time, follower.started, follower.time, work.time)
            else:
                seen = []
                engine.on_complete(marker, lambda e: seen.append(e.time))
                engine.advance()
                rows[backend] = (marker.time, tuple(seen), work.time, engine.now)
            close(engine)
        for backend in NATIVE:
            assert rows[backend] == rows["python"], (mode, backend)


@pytest.mark.parametrize(("count", "units"), ((4097, 1), (4097, 64)))
def test_full_completion_buffer_continues_at_the_same_instant(count, units):
    for backend in BACKENDS:
        engine, events = uniform_engine(backend, count, units)
        engine.advance(horizon=1.0)
        assert (engine.now, engine.completed_count, engine.advance_steps) == (1.0, count, 1)
        assert all(e.started == 0.0 and e.time == 1.0 for e in events)
        engine.advance(horizon=1.0)
        assert engine.completed_count == count and engine.now == 1.0
        close(engine)


def test_output_drains_respect_every_public_advance_boundary():
    names = [f"r{i}" for i in range(32)]
    for backend in BACKENDS:
        eng, evs = uniform_engine(backend, 4097)
        eng.advance(horizon=0.0)
        assert (eng.now, eng.advance_steps) == (0.0, 0) and all(e.started == 0.0 and not e.done for e in evs)
        close(eng)
        eng = Engine([Resource("r", 4097.0)], backend=backend)
        zero = [eng.submit(f"z{i}", {}) for i in range(4097)]
        eng.advance(horizon=0.0)
        assert (eng.completed_count, zero[-1].time, eng._tasks) == (4097, 0.0, {})
        close(eng)
        eng, evs = uniform_engine(backend, 4097)
        assert eng.advance(until=(evs[0],)) == (evs[0],) and eng.completed_count == 4097
        close(eng)
        eng, wave = uniform_engine(backend, 64)
        tail = eng.submit("b", {"r": 1.0}, after=(wave[0],))
        eng.advance(until=(wave[0],))
        assert (eng.now, eng.completed_count, tail.started, tail.time) == (1.0, 64, 1.0, None)
        close(eng)
        eng, wave = uniform_engine(backend, 4097)
        eng.advance(max_steps=1)
        assert (eng.now, eng.advance_steps, eng.completed_count) == (1.0, 1, 4097)
        close(eng)
        eng, wave = uniform_engine(backend, 4097)
        tail = eng.submit("b", {"r": 1.0}, after=(wave[0],))
        with pytest.raises(ExecutionError, match="execution_step_budget_exhausted"):
            eng.advance(max_steps=1)
        assert (eng.completed_count, tail.started, tail.time) == (4097, 1.0, None)
        close(eng)
        eng = Engine([Resource(n, 1025.0) for n in names], backend=backend)
        for i in range(1025):
            eng.submit(f"t{i}", dict.fromkeys(names, 1.0))
        eng.advance(horizon=1.0)
        assert (eng.advance_steps, eng.completed_count, dict(eng.serviced)) == (1, 1025, dict.fromkeys(names, 1025.0))
        close(eng)


# -- arithmetic failures publish one coherent prefix -------------------------
@pytest.mark.parametrize("warm", (0, 63))
def test_subnormal_service_rate_is_a_shared_error(warm):
    for backend in BACKENDS if warm == 0 else NATIVE:
        engine = Engine([Resource("r", 5e-324, units=2)], backend=backend)
        for i in range(warm):
            engine.submit(f"w{i}", {})
        event = engine.submit("a", {"r": 5e-324}, placement={"r": (0,)})
        with pytest.raises(ExecutionError, match="service_rate_underflow:r"):
            engine.advance()
        assert engine.now == 0.0 and not event.done
        close(engine)


@pytest.mark.parametrize(("kind", "oracle"), (
    ("underflow", (1.0, 0.5, 1.0, 64, 2, {"r": 0.5, "x": 0.0}, {"r": 0.5, "x": 1.0})),
    ("zero-coupled-scale", (1.0, 0.5, 1.0, 64, 2, {"r": 0.5, "x": 0.0}, {"r": 0.5, "x": 0.0}))))
def test_late_arithmetic_failure_keeps_the_accepted_prefix(kind, oracle):
    tiny = kind == "underflow"
    for backend in BACKENDS:
        engine = Engine([Resource("r", 1.0), Resource("x", 5e-324 if tiny else 1e308)],
                        backend=backend)
        for i in range(63):
            engine.submit(f"w{i}", {})
        good = engine.submit("good", {"r": 0.5})
        if tiny:
            engine.submit("a", {"x": 5e-323})
        late = engine.submit("late", {"x": 5e-323 if tiny else 5e-324},
                             release=1.0, coupled=not tiny)
        expected = ExecutionError if tiny else (ZeroDivisionError if backend == "python"
                                                else KernelFault)
        with pytest.raises(expected) as info:
            engine.advance()
        if expected is KernelFault:
            assert info.value.has_prefix and info.value.prefix == (1.0, 0, 2, 1, 1)
        assert (engine.now, good.time, late.started, engine.completed_count,
                engine.advance_steps, dict(engine.serviced), dict(engine.busy_ms)) == oracle
        close(engine)


def test_step_zero_failure_exposes_no_prefix():
    for backend in BACKENDS:
        engine = Engine([Resource("r", 1.0), Resource("x", 1e308)], backend=backend)
        for i in range(63):
            engine.submit(str(i), {})
        late = engine.submit("late", {"x": 5e-324}, coupled=True)
        with pytest.raises(ZeroDivisionError if backend == "python" else KernelFault) as info:
            engine.advance()
        if backend != "python":
            assert not info.value.has_prefix and info.value.prefix is None
        assert (engine.now, engine.advance_steps, late.started) == (0.0, 0, 0.0)
        close(engine)


# -- exact-builtin eligibility & Stage.metadata decline table ----------------
class _EqualMeta(type):
    calls: list = []

    def __eq__(cls, other):
        _EqualMeta.calls.append(other)
        return True

    __hash__ = type.__hash__


class _Sneaky(metaclass=_EqualMeta):
    pass


def _nested(depth, leaf=1):
    for _ in range(depth):
        leaf = (leaf,)
    return leaf


_ENUM_V = enum.Enum("_K", "A").A
_INTENUM_V = enum.IntEnum("_L", "A").A
_NTUPLE_V = typing.NamedTuple("_P", [("a", int)])(1)
_DCLASS_V = dataclasses.make_dataclass("_F", [("a", int)], frozen=True)(1)
INERT = (None, True, 1, 1.0, "x", b"x", (), (1, (2.0, None)), _nested(7))
NOT_INERT = (
    frozenset(), frozenset({"p"}), (frozenset({"p"}),), (1, (2.0, (frozenset(),))),
    type("I", (int,), {})(1), type("S", (str,), {})("x"), type("F", (float,), {})(1.5),
    type("Fs", (frozenset,), {})({1}), [1], {"a": 1}, {1}, bytearray(b"x"),
    _ENUM_V, _INTENUM_V, _NTUPLE_V, _DCLASS_V, types.MappingProxyType({"a": 1}),
    _Sneaky(), Token(), _nested(8),
)


def test_inert_builtins_are_non_weakrefable_and_never_dispatch_metaclass_eq():
    _EqualMeta.calls.clear()
    for value in INERT:
        assert tables._is_inert_builtin(value), value
        with pytest.raises(TypeError):
            weakref.ref(value)
    for value in NOT_INERT:
        assert not tables._is_inert_builtin(value), value
    assert _EqualMeta.calls == []


DECLINED_METADATA = (
    {"v": _DCLASS_V}, {"v": _ENUM_V}, {"v": _INTENUM_V}, {"v": _NTUPLE_V},
    {"v": types.MappingProxyType({})}, {"v": [1]}, {"v": {1}},
    {"v": frozenset({"p"})}, {frozenset(): 1}, {"v": (1, (frozenset(),))},
    {"v": {"inner": 1}}, {"v": (1, bytearray(b"x"))}, {"v": Token()},
    {Token(): 1}, {"v": _nested(8)}, type("D", (dict,), {})(layer=0),
)


@pytest.mark.parametrize("metadata", DECLINED_METADATA, ids=repr)
def test_non_builtin_metadata_declines_before_native_integration(metadata):
    engine = Engine([Resource("r", 64.0)], backend="python", trace_limit=0)
    task = engine._tasks[engine.submit("plain", {"r": 1.0}).sequence]
    assert tables.pure_task(engine, task) and not tables.callback_capable(task, engine)
    changed = dataclasses.replace(task, metadata=metadata)
    assert not tables._eligible_metadata(metadata)
    assert not tables._stable_task_payloads(changed, engine)
    assert not tables.pure_task(engine, changed)
    assert tables.callback_capable(changed, engine)
    assert tables.task_flags(changed, engine) & tables.F_CALLBACK
    record = {"task": "plain", "owner": "", "start_ms": 0.0, "end_ms": 1.0,
              "demand": {"r": 1.0}, "serviced": {"r": 1.0}, "metadata": metadata,
              "placement": {}, "coupled": False}
    assert not tables.is_inert_trace_entry(record)


def test_flat_builtin_metadata_stays_native_and_occurrence_declines():
    flat = {"layer": 0, "op": "qkv", "ratio": 0.5, "flag": True, "raw": b"x",
            "none": None, "shape": (1, (2, 3)), "deep": _nested(7)}
    results = {}
    for backend in BACKENDS:
        engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
        events = [engine.submit(f"t{i}", {"r": w}, caps={"r": 1.0}, metadata=flat)
                  for i, w in enumerate(FRONTIER_WORK)]
        engine.advance()
        results[backend] = ([e.time for e in events], engine.serviced["r"],
                            engine.accel_status()["native_steps"])
        task = engine._tasks[engine.submit("occ", {"r": 1.0},
                                           metadata={"occurrence": "op:0"}).sequence]
        assert tables._stable_task_payloads(task, engine) and not tables.pure_task(engine, task)
        assert tables.task_flags(task, engine) & tables.F_CALLBACK
        close(engine)
    assert results["python"][1] == 189.0
    for backend in NATIVE:
        assert results[backend][:2] == results["python"][:2] and results[backend][2] > 0


SUBCLASS_FIELDS = (
    ("name", type("S", (str,), {})("root")), ("owner", type("S", (str,), {})("owner")),
    ("work", {type("S", (str,), {})("r"): 1.0}), ("caps", {type("S", (str,), {})("r"): 1.0}),
    ("caps", {"r": type("F", (float,), {})(1.0)}),
    ("placement", {type("S", (str,), {})("r"): (0,)}),
    ("placement", {"r": (type("I", (int,), {})(0),)}),
    ("release", type("F", (float,), {})(0.0)), ("latency", type("F", (float,), {})(1.0)),
    ("coupled", type("I", (int,), {})(1)), ("priority", type("I", (int,), {})(1)),
    ("cohort_size", type("I", (int,), {})(2)),
    ("memory_bindings", {"unused": {"owner": Token()}}),
    ("memory_pools", frozenset({"p"})),
)


@pytest.mark.parametrize(("field", "value"), SUBCLASS_FIELDS)
def test_every_retained_task_field_is_audited_by_type_identity(field, value):
    engine = Engine([Resource("r", 64.0)], backend="python")
    kwargs = {"caps": {"r": 1.0}, field: value}
    task = engine._tasks[engine.submit(
        kwargs.pop("name", "root0"), kwargs.pop("work", {"r": 1.0}), **kwargs).sequence]
    assert not tables._stable_task_payloads(task, engine) and not tables.pure_task(engine, task)


def test_memory_ledger_pool_sentinels_and_aliased_containers():
    engine = Engine([Resource("r", 64.0)], backend="python")
    plain = engine._tasks[engine.submit("p", {"r": 1.0}).sequence]
    snap = engine._tasks[engine.submit("s", {"r": 1.0}, metadata={"a": 0}).sequence]
    assert not tables._stable_task_payloads(
        dataclasses.replace(plain, memory_ledger=object()), engine)
    assert plain.memory_pools is ENGINE_EMPTY_FROZENSET
    assert snap.memory_pools is engine._memory_pools_frozen
    assert tables.pure_task(engine, plain) and tables.pure_task(engine, snap)
    assert not tables._inert_pool_set(engine._memory_pools_frozen)
    for foreign in (frozenset(), frozenset({"p"}), ("p",), frozenset({1})):
        assert not tables._inert_pool_set(foreign, engine), foreign
    finalizer = weakref.finalize(ENGINE_EMPTY_FROZENSET, lambda: None)
    try:
        assert not tables._inert_pool_set(ENGINE_EMPTY_FROZENSET, engine)
    finally:
        finalizer.detach()
    engine.bind_memory(object(), pools=("hbm",))
    assert tables._inert_pool_set(engine._memory_pools_frozen, engine)
    for ctor in (list, dict, set, bytearray):
        assert not tables.metadata_is_stable({"payload": ctor()})
        assert not tables.metadata_is_stable({"payload": _nested(9, ctor())})
    assert tables.metadata_is_stable({"token": Token(), "shape": (1, 128)})


@pytest.mark.parametrize(("kwargs", "reason"), (
    ({"priority": 2 ** 63}, "wide_integer"), ({"cohort_size": 2 ** 64 + 1}, "wide_integer"),
    ({"cohort_size": 2 ** 64 - 1}, "wide_integer"),
    ({"cohort_size": 2 ** 64 - 2, "priority": 2 ** 63 - 1}, None)))
def test_unrepresentable_integers_and_mixed_fair_priorities_decline(kwargs, reason):
    engine = Engine([Resource("r", 1.0)], backend="c")
    event = engine.submit("e", {"r": 1.0}, **kwargs)
    engine.advance(until=(event,))
    status = engine.accel_status()
    assert event.time == 1.0 and status["fallback_reason"] == reason
    assert (status["native_steps"] > 0) == (reason is None)
    close(engine)
    mixed = Engine([Resource("r", 4.0)], backend="c")
    events = [mixed.submit(f"t{i}", {"r": 1.0}, priority=i % 2) for i in range(8)]
    mixed.advance()
    assert mixed.accel_status()["fallback_reason"] == "mixed_priority"
    assert mixed.accel_status()["native_steps"] == 0 and all(e.done for e in events)
    close(mixed)


# -- coupled proportional placement & resource/unit ceilings ----------------
@pytest.mark.parametrize("sku", SKUS)
@pytest.mark.parametrize("extra", (0, 1), ids=("proportional", "skewed"))
def test_coupled_placement_must_be_proportional_to_owner_counts(sku, extra):
    cube, vector = sku_resources(sku)[:2]
    width = cube.units // 2
    aiv = width * vector.units // cube.units + extra
    placement = {"cube": tuple(range(width)), "vector": tuple(range(aiv))}
    proportional = aiv * cube.units == width * vector.units
    results = {}
    for backend in BACKENDS:
        engine = Engine([cube, vector], backend=backend)
        events = [engine.submit(f"c{i}", {"cube": cube.rate, "vector": vector.rate},
                                coupled=True, placement=placement) for i in range(64)]
        assert tables.pure_task(engine, engine._tasks[events[0].sequence]) is proportional
        engine.advance()
        results[backend] = ([e.time for e in events], engine.accel_status()["native_steps"])
        close(engine)
    for backend in NATIVE:
        assert max(ulps(a, b) for a, b in zip(results["python"][0],
                                              results[backend][0], strict=True)) <= ULP_BOUND
        assert (results[backend][1] > 0) is proportional


def test_arena_resource_and_unit_ceilings_decline_one_past_limit():
    for count, reason in ((2048, None), (2049, "resource_count")):
        eng = Engine([Resource(f"r{i}", 1.0) for i in range(count)], backend="c")
        [eng.submit(f"t{i}", {f"r{i}": 1.0}) for i in range(count)]
        eng.advance()
        assert eng.accel_status()["fallback_reason"] == reason and (eng.accel_status()["native_steps"] > 0) == (reason is None)
        close(eng)
    for units, reason in ((128, None), (129, "resource_shape")):
        eng = Engine([Resource("r", float(units), units=units)], backend="c")
        [eng.submit(f"t{i}", {"r": 1.0}, placement={"r": tuple(range(units))}) for i in range(64)]
        eng.advance()
        assert eng.accel_status()["fallback_reason"] == reason and (eng.accel_status()["native_steps"] > 0) == (reason is None)
        close(eng)


# -- raw engine-arena/2 ABI -------------------------------------------------
def test_abi_probe_and_concurrent_contexts_are_independent():
    library = native_library()
    assert int(library.msim_abi_version()) == ABI_VERSION
    probe = (ctypes.c_uint64 * 16)()
    assert int(library.msim_abi_probe(probe, 16)) == 0 and tuple(int(x) for x in probe) == PROBE_GOLDEN
    assert int(library.msim_abi_probe(probe, 4)) != 0
    arenas = [open_arena([cap]) for cap in (256.0, 512.0)]
    for arena, cap in zip(arenas, (256.0, 512.0), strict=True):
        assert len(arena.submit([arena_row(seq=i + 1) for i in range(128)])) == 128
        assert arena.run(0.0, INF, INF, 100000)[3] == 128
        busy, serv = arena.counters()
        assert abs(serv[0] - 128.0) < 1e-12 and abs(busy[0] - 128.0 / cap) < 1e-12
        arena.close()

    def run(capacity):
        eng = Engine([Resource("r", capacity)], backend="c")
        eng.advance(until=tuple(eng.submit(f"t{i}", {"r": 1.0}) for i in range(128)))
        out = (eng.accel_status()["native_steps"] > 0, eng.serviced["r"], eng.busy_ms["r"] * capacity)
        close(eng)
        return out

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(run, (256.0, 512.0, 256.0, 512.0))) == [(True, 128.0, 128.0)] * 4


def test_kernel_64bit_sequences_and_unrepresentable_row_range_errors():
    for seq in (2 ** 31 + 100, 2 ** 63 + 7, 2 ** 64 - 1):
        arena = open_arena([100.0])
        arena.submit([arena_row(seq=seq, cohort=2 ** 64 - 2)])
        arena.run(0.0, INF, INF, 100)
        assert arena.completions(1)[0][0] == seq
        arena.close()
    arena = open_arena()
    for row in (arena_row(prio=2 ** 63), arena_row(cohort=2 ** 64 + 1), arena_row(seq=2 ** 64)):
        with pytest.raises(KernelFault) as info:
            arena.submit([row])
        assert info.value.code == tables.E_RANGE
    arena.close()


@pytest.mark.parametrize(("version", "comp", "now", "code"), (
    (ABI_VERSION + 1, 3 * 4096, 0.0, tables.E_ABI),
    (ABI_VERSION, 0, 0.0, tables.E_FULL),
    (ABI_VERSION, 3 * 4096, float("nan"), tables.E_VALUE)))
def test_malformed_run_inputs_fail_before_mutation(version, comp, now, code):
    arena = open_arena()
    arena.submit([arena_row()])
    arena.hdr[0], arena.hdr[1], arena.hdr[2] = version, 0, 10
    arena.io[0], arena.io[1], arena.io[3] = now, INF, INF
    rc = arena.library.msim_ctx_run(arena.ctx, arena.hdr, 16, arena.st, 8, arena.io, 8,
                                    arena.comp, comp, arena.svc, 3 * 32768, arena.started, 2 * 4096)
    assert rc != 0 and int(arena.st[1]) == code and arena.run(0.0, INF, INF, 100)[0::3] == (0.1, 1)
    arena.close()


def test_overlapping_links_and_rejected_link_batch_are_atomic():
    arena = open_arena()
    rows = (ctypes.c_uint64 * (65 * 12))()
    for i in range(65):
        rows[i * 12:(i + 1) * 12] = [i + 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
    links = (ctypes.c_uint64 * 6)(0, 0, tables.bits(1.0), tables.bits(INF), 0, 0)
    slots = (ctypes.c_int32 * 65)()
    rc = arena.library.msim_ctx_submit(arena.ctx, rows, 65, 12, links, 1, 6, None,
                                       0, None, None, 0, slots, 65, arena.st, 8)
    assert rc != 0 and int(arena.st[1]) == tables.E_RANGE
    arena.submit([arena_row()])
    assert arena.run(0.0, INF, INF, 100)[0::3] == (0.1, 1)
    arena.close()
    observed = []
    for poisoned in (False, True):
        arena = open_arena([1.0])
        s = arena.submit([arena_row(seq=1), arena_row(seq=2, waiting=2)])
        if poisoned:
            with pytest.raises(KernelFault) as info:
                arena.link([(s[0], s[1], 2), (s[0], 999999, 3)])
            assert info.value.code == tables.E_RANGE
        arena.link([(s[0], s[1], 2)])
        now, _b, steps, ncomp, _ns = arena.run(0.0, INF, INF, 100)
        observed.append((now, steps, ncomp, tuple(arena.completions(ncomp))))
        arena.close()
    assert observed[0] == observed[1] and observed[0][0::2] == (1.0, 1)


@pytest.mark.parametrize(("count", "nres", "trigger"), ((4097, 1, 4095), (1025, 32, 1023)))
def test_pending_hold_survives_raw_wire_output_split(count, nres, trigger):
    arena = open_arena([float(count)] * nres)
    try:
        links = [(i, 1.0, math.inf, ()) for i in range(nres)]
        rows = [(i + 1, tables.F_CALLBACK if i == trigger else 0, 0, 1, 0, 0.0, 0.0, 0.0, links)
                for i in range(count)] + [(count + 1, 0, 0, 1, 1, 0.0, 0.0, 0.0, links)]
        slots = arena.submit(rows)
        arena.link([(slots[trigger], slots[-1], count + 1)])
        while arena.run(0.0, 0.0, math.inf, 0)[1] == tables.B_OUTPUT:
            pass
        split = arena.run(0.0, 1.0, math.inf, 1)
        assert split[1] == tables.B_OUTPUT and int(arena.hdr[3]) == 1
        assert split[3] == (4096 if nres == 1 else 1024) and split[4] == 0
        cont = arena.run(1.0, 1.0, math.inf, 0)
        assert cont[1] in (tables.B_CALLBACK, tables.B_HORIZON) and int(arena.hdr[3]) == 1
        assert cont[3] == 1 and cont[4] == 0
        admit = arena.run(1.0, 1.0, math.inf, 0)
        assert int(arena.hdr[3]) == 0 and admit[3] == 0
        assert [seq for seq, _t in arena.starts(admit[4])] == [count + 1]
    finally:
        arena.close()
