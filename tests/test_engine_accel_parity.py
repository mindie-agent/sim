"""Adversarial regression suite for the engine-arena/2 C11 accelerator.

[Role] Exercise the per-Engine kernel context, its hardened ABI, the incremental
  pure-DAG epoch and the Astra findings (R1-R15) the runtime owns.
[Invariants] Every test uses an isolated cache/HOME and never touches the real
  user cache.  Native behaviour is compared against the canonical Python Engine
  on the same fixture; a kernel fault is asserted, never replayed.
[Timeouts & Locks] No sleeps; subprocess calls carry explicit timeouts.
[Side-Effect Policy] Writes only temporary directories created by the fixture.
"""
from __future__ import annotations

import concurrent.futures
import gc
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "mindie_sim" / "prediction" / "runtime"

sys.path.insert(0, str(ROOT))

from mindie_sim.prediction.runtime import accel_bridge, accel_build  # noqa: E402
from mindie_sim.prediction.runtime.engine import (  # noqa: E402
    Engine, ExecutionError, Resource,
)

LINE_CAPS = {
    "_engine_accel.c": 750,
    "_engine_accel_core.h": 750,
    "_engine_accel_rates.h": 750,
    "_engine_accel_abi.h": 750,
    "_engine_accel_submit.h": 750,
    "accel_tables.py": 750,
    "accel_bridge.py": 750,
    "accel_build.py": 750,
    "accel_sync.py": 750,
    "engine.py": 2100,
    "frontier.py": 200,
}


def _library():
    status = accel_build.build_accel()
    assert status["ok"], status.get("detail") or status.get("reason")
    return accel_bridge.load_library(status)


def test_modules_obey_line_and_encoding_caps():
    tags = ("[Role]", "[Invariants]", "[Timeouts & Locks]",
            "[Side-Effect Policy]")
    for name, cap in LINE_CAPS.items():
        data = (RUNTIME / name).read_bytes()
        assert b"\r" not in data, name
        assert data.endswith(b"\n"), name
        assert len(data.splitlines()) < cap, (name, len(data.splitlines()))
        if name.endswith(".py"):
            head = "\n".join(data.decode().splitlines()[:30])
            assert all(tag in head for tag in tags), name


def test_concurrent_engines_have_disjoint_contexts():
    _library()  # prewarm the one process-local immutable library
    errors = []
    contenders = 128

    def run(thread_id: int, capacity: float) -> None:
        engine = Engine([Resource("r", capacity)], backend="c")
        events = [engine.submit(f"t{thread_id}-{i}", {"r": 1.0})
                  for i in range(contenders)]
        engine.advance(until=tuple(events))
        status = engine.accel_status()
        if status["native_steps"] <= 0:
            errors.append((thread_id, "no_native_steps", status["native_steps"]))
        if abs(engine.serviced["r"] - contenders) > 1e-9:
            errors.append((thread_id, engine.serviced["r"]))
        expected = contenders / capacity
        if abs(engine.busy_ms["r"] - expected) > 1e-9:
            errors.append((thread_id, "busy", engine.busy_ms["r"], expected))

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run, index, 512.0 if index % 2 else 256.0)
                   for index in range(4)]
        for future in futures:
            future.result()
    assert errors == []


def test_long_periodic_sequence_then_tail_via_python_reference():
    from mindie_sim.prediction.program import PipelinePhase, PipelineTemplate, Stage
    from mindie_sim.prediction.scheduling import start_pipeline_template

    engine = Engine([Resource("r", 100.0)], backend="python")
    template = PipelineTemplate(
        owner=0, slots=2,
        phases=(PipelinePhase("main",
                              (Stage("work", {"r": 100.0},
                                     previous=("work",)),),
                2 ** 31 + 100),))
    done = start_pipeline_template(engine, template, "long",
                                   engine.submit("ready"))
    engine.advance(until=(done,), max_steps=100000)
    tail = engine.submit("tail", {"r": 100.0})
    assert tail.sequence > 2 ** 31
    engine.advance(until=(tail,), max_steps=10)
    assert tail.done and tail.time is not None


def test_conserved_work_is_bit_exact_across_resources():
    resources = [Resource("mte", 64.0), Resource("cube", 128.0, units=4),
                 Resource("vec", 32.0, units=2, policy="fifo")]
    results = {}
    for backend in ("python", "c"):
        engine = Engine(resources, backend=backend)
        events = [engine.submit(
            f"t{index}",
            {"mte": 2.0 + index % 5, "cube": 4.0 + index % 3,
             "vec": 1.0 + index % 2}) for index in range(48)]
        engine.advance(until=tuple(events))
        results[backend] = (engine.now, dict(engine.serviced), dict(engine.busy_ms),
                            {event.name: event.time for event in events})
    py, native = results["python"], results["c"]
    assert native[0] == py[0]
    assert native[3] == py[3]
    for name, value in py[1].items():
        scale = max(1.0, abs(value), abs(native[1][name]))
        assert abs(native[1][name] - value) / scale <= 1e-15, name
    for name, value in py[2].items():
        scale = max(1.0, abs(value), abs(native[2][name]))
        assert abs(native[2][name] - value) / scale <= 1e-15, name


def test_coupled_equal_placement_tuples_are_interned_by_value():
    results = {}
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("g", 64.0, units=64)], backend=backend)
        units = tuple(range(64))
        events = [engine.submit(f"c{index}", {"g": 1.0},
                                placement={"g": units}, coupled=True)
                  for index in range(257)]
        engine.advance(until=tuple(events))
        results[backend] = (
            tuple(event.time for event in events), dict(engine.serviced),
            engine.accel_status()["native_steps"])
    expected = 257.0 / 64.0
    assert all(abs(when - expected) < 1e-9 for when in results["c"][0])
    assert results["c"][1] == results["python"][1]
    assert results["auto"][0] == results["python"][0]
    assert results["auto"][1] == results["python"][1]
    assert abs(results["c"][1]["g"] - 257.0) < 1e-9
    assert results["c"][2] > 0 and results["auto"][2] > 0


def test_subnormal_rate_underflow_is_a_shared_error():
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("r", 5e-324)], backend=backend)
        a = engine.submit("a", {"r": 5e-324})
        b = engine.submit("b", {"r": 5e-324})
        with pytest.raises(ExecutionError, match="service_rate_underflow:r"):
            engine.advance(horizon=1.0)
        assert engine.now == 0.0
        assert not a.done and not b.done


def test_auto_tiny_workload_has_zero_native_calls():
    engine = Engine([Resource("r", 1000.0)], backend="auto")
    events = [engine.submit(f"t{index}", {"r": 1.0}) for index in range(8)]
    engine.advance(until=tuple(events))
    status = engine.accel_status()
    assert status["native_calls"] == 0
    assert status["accel_backend"] in {"python", "python_fallback"}


def test_python_backend_performs_zero_discovery(monkeypatch):
    engine = Engine([Resource("r", 10.0)], backend="python")
    assert engine.accel_status()["accel_backend"] == "python"
    assert engine.accel_status()["fallback_reason"] == "requested_python"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("python backend touched the toolchain")

    monkeypatch.setattr(accel_build, "discover_compiler", forbidden)
    assert Engine([Resource("r", 10.0)], backend="python") is not None


def test_source_and_counterexamples_execute_clean_under_asan_ubsan(tmp_path):
    source = RUNTIME / "_engine_accel.c"
    harness = ROOT / "tests" / "accel_sanitizer_harness.c"
    cc = (accel_build.discover_compiler()[0] if os.name == "nt"
          else (shutil.which("clang") or accel_build.discover_compiler()[0]))
    binary = tmp_path / ("accel_sanitizer.exe" if os.name == "nt" else "accel_sanitizer")
    san = () if os.name == "nt" else ("-fsanitize=address,undefined", "-fno-sanitize-recover=all")
    libm = () if os.name == "nt" else ("-lm", "-pthread")
    base_cmd = [cc, "-std=c11", "-O1", "-g", "-ffp-contract=off", "-I", str(RUNTIME),
                str(source), str(harness), "-o", str(binary), *libm]
    completed = subprocess.run([*base_cmd[:4], *san, *base_cmd[4:]], capture_output=True,
                               text=True, timeout=240, check=False)
    if completed.returncode != 0 and san:
        completed = subprocess.run(base_cmd, capture_output=True, text=True, timeout=240, check=False)
    assert completed.returncode == 0, completed.stderr[-2000:]
    environment = dict(os.environ)
    environment["ASAN_OPTIONS"] = "detect_leaks=0:halt_on_error=1"
    environment["UBSAN_OPTIONS"] = "halt_on_error=1:print_stacktrace=1"
    executed = subprocess.run([str(binary)], capture_output=True, text=True,
                              timeout=240, check=False, env=environment)
    output = executed.stdout + executed.stderr
    assert executed.returncode == 0, output[-3000:]
    assert "sanitizer_harness: ok" in executed.stdout
    assert "AddressSanitizer" not in output
    assert "runtime error" not in output
    strict = subprocess.run(
        [cc, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
         "-ffp-contract=off", "-I", str(RUNTIME), "-c", str(source),
         "-o", str(tmp_path / "accel.o")],
        capture_output=True, text=True, timeout=180, check=False)
    assert strict.returncode == 0, strict.stderr[-2000:]
    harness_strict = subprocess.run(
        [cc, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
         "-ffp-contract=off", "-I", str(RUNTIME), "-c", str(harness),
         "-o", str(tmp_path / "harness.o")],
        capture_output=True, text=True, timeout=180, check=False)
    assert harness_strict.returncode == 0, harness_strict.stderr[-2000:]


def test_experimental_epoch_matches_python_on_a_pure_frontier():
    resources = [Resource("mte", 32.0), Resource("vec", 64.0, units=2)]
    payload = [((index % 7) + 1.0, (index % 5) + 1.0) for index in range(24)]
    results = {}
    for backend in ("python", "c"):
        engine = Engine(resources, backend=backend)
        events = [engine.submit(f"p{index}", {"mte": pair[0], "vec": pair[1]})
                  for index, pair in enumerate(payload)]
        engine.advance(until=tuple(events))
        results[backend] = (engine.now, dict(engine.busy_ms), dict(engine.serviced),
                            {event.name: event.time for event in events},
                            engine.accel_status()["native_steps"])
    assert results["c"][0] == results["python"][0]
    assert results["c"][3] == results["python"][3]
    assert results["c"][1] == results["python"][1]
    assert results["c"][2] == results["python"][2]
    assert results["c"][4] > 0


def test_experimental_epoch_fifo_priority_and_cohort():
    results = {}
    for backend in ("python", "c"):
        engine = Engine([Resource("f", 1.0, policy="fifo")], backend=backend)
        first = engine.submit("first", {"f": 1.0})
        second = engine.submit("second", {"f": 1.0})
        engine.advance(until=(first, second))
        results[backend] = (first.time, second.time, dict(engine.serviced))
    assert results["c"] == results["python"]


def test_start_buffer_exhaustion_preserves_every_started_event():
    for backend in ("c", "auto"):
        engine = Engine([Resource("r", 4097.0)], backend=backend)
        events = [engine.submit(f"t{index}", {"r": 1.0})
                  for index in range(4097)]
        engine.advance()
        assert all(event.started is not None for event in events)
        assert all(event.done for event in events)
        assert engine.now == 1.0
        assert engine.accel_status()["native_steps"] > 0, backend


def test_wide_integer_declines_native_and_uint64_cohort_runs():
    engine = Engine([Resource("r", 1.0)], backend="c")
    event = engine.submit("wide", {"r": 1.0}, priority=2 ** 63)
    engine.advance(until=(event,))
    status = engine.accel_status()
    assert status["native_steps"] == 0
    assert status["fallback_reason"] == "wide_integer"
    assert event.done and event.time == 1.0

    engine = Engine([Resource("r", 1.0)], backend="c")
    event = engine.submit("wide-cohort", {"r": 1.0},
                          cohort_size=2 ** 64 - 2, priority=2 ** 63 - 1)
    engine.advance(until=(event,))
    assert engine.accel_status()["native_steps"] > 0
    assert event.time == 1.0

    engine = Engine([Resource("r", 1.0)], backend="c")
    event = engine.submit("too-wide", {"r": 1.0}, cohort_size=2 ** 64 + 1)
    engine.advance(until=(event,))
    assert engine.accel_status()["fallback_reason"] == "wide_integer"
    assert event.time == 1.0

    engine = Engine([Resource("r", 1.0)], backend="c")
    event = engine.submit("sentinel-cohort", {"r": 1.0}, cohort_size=2 ** 64 - 1)
    engine.advance(until=(event,))
    assert engine.accel_status()["fallback_reason"] == "wide_integer"
    assert event.time == 1.0


def test_incremental_placement_matches_python():
    results = {}
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("r", 2.0, units=2)], backend=backend)
        first = engine.submit("A", {"r": 2.0}, placement={"r": (0,)})
        engine.advance(horizon=0.5)
        second = engine.submit("B", {"r": 1.0}, placement={"r": (1,)})
        engine.advance()
        results[backend] = (
            first.time, second.time, dict(engine.serviced),
            dict(engine.busy_ms), engine.now,
            engine.accel_status()["native_steps"])
    assert results["c"][:5] == results["python"][:5]
    assert results["auto"][:5] == results["python"][:5]
    assert (results["python"][0], results["python"][1]) == (2.0, 1.5)
    assert results["c"][5] > 0


def test_mixed_frontier_latency_handback_matches_python():
    results = {}
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("r", 1.0)], backend=backend)
        sleep = engine.submit("sleep", {}, latency=10.0)
        engine.advance(horizon=1.0)
        seen = []
        work = engine.submit("work", {"r": 1.0},
                             on_complete=lambda event: seen.append(event.time))
        engine.advance()
        results[backend] = (
            engine.now, sleep.time, work.time, tuple(seen),
            engine.accel_status()["native_steps"])
    assert results["python"][:4] == (10.0, 10.0, 2.0, (2.0,))
    for backend in ("c", "auto"):
        assert results[backend][:4] == results["python"][:4]
    assert results["c"][4] > 0


def test_trace_end_uses_the_completion_moment():
    results = {}
    for backend in ("python", "c"):
        engine = Engine([Resource("r", 1.0)], backend=backend, trace_limit=16)
        engine.submit("w1", {"r": 1.0})
        engine.submit("w2", {"r": 2.0})
        engine.advance()
        results[backend] = [(row["task"], row["end_ms"]) for row in engine.trace]
        if backend == "c":
            assert engine.accel_status()["native_steps"] > 0
    assert results["c"] == results["python"] == [("w1", 2.0), ("w2", 3.0)]


def test_native_contexts_are_freed_by_gc(monkeypatch):
    library = _library()
    calls = []
    original = library.msim_ctx_free

    def counted(context):
        calls.append(context)
        return original(context)

    monkeypatch.setattr(library, "msim_ctx_free", counted)
    gc.collect()
    baseline = len(calls)

    def build():
        engines = []
        for _ in range(3):
            engine = Engine([Resource("r", 10.0)], backend="c")
            events = [engine.submit(f"t{index}", {"r": 1.0})
                      for index in range(64)]
            engine.advance(until=tuple(events))
            assert engine.accel_status()["native_steps"] > 0
            engines.append(engine)
        return engines

    engines = build()
    del engines
    gc.collect()
    assert len(calls) - baseline == 3


def test_auto_load_failure_falls_back_on_late_submission(monkeypatch):
    _library()

    def unavailable(status):
        raise accel_bridge.AccelUnavailable("library_load_failed:injected", status)

    monkeypatch.setattr(accel_bridge, "load_library", unavailable)
    with pytest.raises(accel_bridge.AccelUnavailable):
        Engine([Resource("r", 10.0)], backend="c")
    engine = Engine([Resource("r", 10.0)], backend="auto")
    events = [engine.submit(f"t{index}", {"r": 1.0})
              for index in range(64)]
    engine.advance(until=tuple(events))
    status = engine.accel_status()
    assert status["accel_backend"] in {"python", "python_fallback"}
    assert "injected" in (status["backend_reason"] or "")
    assert status["native_steps"] == 0
    assert all(event.done for event in events)


def test_summary_exposes_native_coverage_counters():
    engine = Engine([Resource("r", 1000.0)], backend="c")
    events = [engine.submit(f"t{index}", {"r": 1.0}) for index in range(64)]
    engine.advance(until=tuple(events))
    summary = engine.summary()
    assert summary["accel_backend"] == "c_native"
    assert summary["native_steps"] > 0
    assert summary["native_calls"] > 0
    assert summary["fallback_reason"] is None


def test_auto_adopts_an_unstarted_pure_burst():
    engine = Engine([Resource("r", 1000.0)], backend="auto")
    events = [engine.submit(f"t{index}", {"r": 1.0}) for index in range(64)]
    engine.advance(until=tuple(events))
    status = engine.accel_status()
    assert status["native_steps"] > 0
    assert all(event.done for event in events)


def test_harness_obeys_the_line_and_encoding_rules():
    tags = ("[Role]", "[Invariants]", "[Timeouts & Locks]",
            "[Side-Effect Policy]")
    paths = sorted((ROOT / "tests").glob("test_engine_accel*.py")) + [ROOT / "tests" / "conftest.py"]
    paths.append(ROOT / "tests" / "accel_sanitizer_harness.c")
    paths.append(ROOT / "tests" / "accel_sanitizer_wide.h")
    for path in paths:
        data = path.read_bytes()
        assert b"\r" not in data, path.name
        assert data.endswith(b"\n"), path.name
        assert len(data.splitlines()) < 750, path.name
        if path.suffix == ".py":
            head = "\n".join(data.decode().splitlines()[:30])
            assert all(tag in head for tag in tags), path.name


def test_auto_dependent_adoption_matches_python_after_handback():
    """F1: adopted predecessors are Events, and the graph hands back cleanly."""
    results = {}
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("r", 1.0)], backend=backend)
        first = engine.submit("A", {"r": 1.0})
        second = engine.submit("B", {"r": 1.0}, after=(first,))
        rest = [engine.submit(f"x{index}", {"r": 1.0}) for index in range(62)]
        engine.advance()
        events = (first, second) + tuple(rest)
        assert all(event.done for event in events), backend
        status = engine.accel_status()
        results[backend] = (
            tuple(event.time for event in events), engine.now,
            status["native_steps"], status["fallback_reason"])
    assert results["c"][:2] == results["python"][:2]
    assert results["auto"][:2] == results["python"][:2]
    assert results["c"][2] > 0 and results["auto"][2] > 0
    assert results["auto"][3] is None


def test_handback_reconciles_dependency_readiness():
    """F2: a native predecessor completion must not strand a Python successor."""
    results = {}
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("r", 1.0)], backend=backend)
        first = engine.submit("A", {"r": 1.0})
        second = engine.submit("B", {"r": 1.0}, after=(first,))
        engine.advance(until=(first,))
        third = engine.submit("D", {"r": 1.0}, on_complete=lambda event: None)
        engine.advance()
        results[backend] = (
            first.time, second.time, third.time, engine.now,
            engine.accel_status()["native_steps"])
    assert results["python"][:4] == (1.0, 3.0, 3.0, 3.0)
    for backend in ("c", "auto"):
        assert results[backend][:4] == results["python"][:4], backend
    assert results["c"][4] > 0


def test_multihop_chain_handback_matches_python():
    """F2: A -> B -> C readiness survives a native-to-Python hand-back."""
    results = {}
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("r", 1.0)], backend=backend)
        first = engine.submit("A", {"r": 1.0})
        second = engine.submit("B", {"r": 1.0}, after=(first,))
        third = engine.submit("C", {"r": 1.0}, after=(second,))
        engine.advance(until=(second,))
        extra = engine.submit("D", {"r": 1.0}, on_complete=lambda event: None)
        engine.advance()
        results[backend] = (first.time, second.time, third.time, extra.time,
                            engine.now, engine.accel_status()["native_steps"])
    for backend in ("c", "auto"):
        assert results[backend][:5] == results["python"][:5], backend
    assert results["python"][:5] == (1.0, 2.0, 4.0, 4.0, 4.0)
    assert results["c"][5] > 0


def test_diamond_handback_matches_python():
    """F2: a diamond's Python waiting counts reconcile after native completion."""
    results = {}
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("r", 1.0)], backend=backend)
        root = engine.submit("A", {"r": 1.0})
        left = engine.submit("B", {"r": 1.0}, after=(root,))
        right = engine.submit("C", {"r": 1.0}, after=(root,))
        join = engine.submit("D", {"r": 1.0}, after=(left, right))
        engine.advance(until=(root,))
        extra = engine.submit("Z", {"r": 1.0}, on_complete=lambda event: None)
        engine.advance()
        results[backend] = (root.time, left.time, right.time, join.time,
                            extra.time, engine.now)
    for backend in ("c", "auto"):
        assert results[backend] == results["python"], backend


def test_adopted_dependency_graph_hands_back_without_stranding():
    """F1/F2: auto adoption of A -> B -> C keeps exact readiness on hand-back."""
    results = {}
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("r", 1.0)], backend=backend)
        first = engine.submit("A", {"r": 1.0})
        second = engine.submit("B", {"r": 1.0}, after=(first,))
        third = engine.submit("C", {"r": 1.0}, after=(second,))
        rest = [engine.submit(f"x{index}", {"r": 1.0}) for index in range(61)]
        engine.advance(until=(first,))
        extra = engine.submit("D", {"r": 1.0}, on_complete=lambda event: None)
        engine.advance()
        events = (first, second, third, extra) + tuple(rest)
        assert all(event.done for event in events), backend
        results[backend] = (tuple(event.time for event in events), engine.now,
                            engine.accel_status()["native_steps"])
    assert results["auto"][:2] == results["python"][:2]
    assert results["c"][:2] == results["python"][:2]
    assert results["auto"][2] > 0 and results["c"][2] > 0


def test_two_thousand_sequential_placements_never_exhaust_the_pool():
    """F3: sequential 64-unit placements grow/reclaim instead of lifetime-failing."""
    reference = None
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("r", 64.0, units=64)], backend=backend)
        if backend != "c":
            for index in range(63):
                engine.submit(f"w{index}", {})
            engine.advance()
        units = tuple(range(64))
        times = []
        for index in range(2000):
            event = engine.submit(f"t{index}", {"r": 1.0},
                                  placement={"r": units})
            engine.advance(until=(event,))
            assert event.done, (backend, index)
            times.append(event.time)
        if backend == "python":
            reference = (engine.now, tuple(times))
            continue
        status = engine.accel_status()
        assert status["fallback_reason"] is None, backend
        assert status["native_steps"] >= 2000, (backend, status["native_steps"])
        assert (engine.now, tuple(times)) == reference, backend


def test_simultaneous_placement_pool_grows_without_overflow():
    """F3: one 128,000-entry batch exceeds the old 65,536 lifetime pool."""
    results = {}
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("r", 64.0, units=64)], backend=backend)
        units = tuple(range(64))
        events = [engine.submit(f"t{index}", {"r": 1.0},
                                placement={"r": units}) for index in range(2000)]
        engine.advance()
        assert all(event.done for event in events), backend
        results[backend] = (engine.now, tuple(event.time for event in events),
                            dict(engine.serviced),
                            engine.accel_status()["fallback_reason"])
    assert results["c"][:3] == results["python"][:3]
    assert results["auto"][:3] == results["python"][:3]
    assert results["c"][3] is None and results["auto"][3] is None


def test_static_task_and_link_caps_decline_to_python():
    """F3: a live frontier beyond MAX_T/MAX_L never reaches a kernel fault."""
    engine = Engine([Resource("r", 1_000_000.0)], backend="c")
    for index in range(65537):
        engine.submit(f"t{index}", {})
    status = engine.accel_status()
    assert status["fallback_reason"] == "capacity_exceeded"
    assert status["accel_backend"] == "python_fallback"
    assert status["native_steps"] == 0
    del engine

    resources = [Resource(f"r{index}", 1000.0) for index in range(32)]
    work = {f"r{index}": 1.0 for index in range(32)}
    engine = Engine(resources, backend="c")
    for index in range(8193):
        engine.submit(f"t{index}", work)
    status = engine.accel_status()
    assert status["fallback_reason"] == "capacity_exceeded"
    assert status["native_steps"] == 0


def test_start_boundary_pure_observation_matches_python():
    """F4: a passive start observer resolves natively at the exact start tick."""
    results = {}
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("r", 1.0)], backend=backend)
        for index in range(63):
            engine.submit(f"w{index}", {})
        work = engine.submit("work", {"r": 1.0})
        marker = engine.start_boundary(work)
        engine.advance()
        results[backend] = (marker.time, marker.started, work.time, engine.now,
                            engine.accel_status()["native_steps"])
    assert results["python"][:4] == (0.0, 0.0, 1.0, 1.0)
    assert results["c"][:4] == results["python"][:4]
    assert results["auto"][:4] == results["python"][:4]
    assert results["c"][4] > 0 and results["auto"][4] > 0


def test_start_boundary_until_dependency_and_callback_match_python():
    """F4: mid-step start causality is exact for until, after=(s,), and callbacks."""
    results = {}
    for mode in ("until", "dependency", "callback"):
        for backend in ("python", "c", "auto"):
            engine = Engine([Resource("r", 1.0)], backend=backend)
            for index in range(63):
                engine.submit(f"w{index}", {})
            work = engine.submit("work", {"r": 1.0})
            marker = engine.start_boundary(work)
            if mode == "until":
                engine.advance(until=(marker,))
                row = (marker.time, work.started, work.time, engine.now)
            elif mode == "dependency":
                follower = engine.submit("follower", {"r": 1.0},
                                         after=(marker,))
                engine.advance()
                row = (marker.time, follower.started, follower.time,
                       work.time, engine.now)
            else:
                seen = []
                engine.on_complete(marker, lambda event: seen.append(event.time))
                engine.advance()
                row = (marker.time, tuple(seen), work.time, engine.now)
            results.setdefault(mode, {})[backend] = row
    for mode, rows in results.items():
        assert rows["c"] == rows["python"], mode
        assert rows["auto"] == rows["python"], mode
    assert results["until"]["python"] == (0.0, 0.0, None, 0.0)
    assert results["dependency"]["python"][:4] == (0.0, 0.0, 2.0, 2.0)
    assert results["callback"]["python"] == (0.0, (0.0,), 1.0, 1.0)


def test_subnormal_zero_frontier_without_horizon_is_shared_error():
    """F5: an all-zero allocated frontier is E_ARITH even with until=None."""
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("r", 5e-324, units=2)], backend=backend)
        event = engine.submit("a", {"r": 5e-324}, placement={"r": (0,)})
        with pytest.raises(ExecutionError, match="service_rate_underflow:r"):
            engine.advance()
        assert engine.now == 0.0
        assert not event.done
    for backend in ("c", "auto"):
        engine = Engine([Resource("r", 5e-324, units=2)], backend=backend)
        for index in range(63):
            engine.submit(f"w{index}", {})
        event = engine.submit("a", {"r": 5e-324}, placement={"r": (0,)})
        with pytest.raises(ExecutionError, match="service_rate_underflow:r"):
            engine.advance()
        assert engine.now == 0.0
        assert not event.done


def test_engine_finalizer_collects_pending_metadata_cycle(monkeypatch):
    """F6: the finalizer retains only (lib, ctx_box), never Engine/Backend/Task."""
    import weakref
    library = _library()
    calls = []
    original = library.msim_ctx_free

    def counted(context):
        calls.append(context)
        return original(context)

    monkeypatch.setattr(library, "msim_ctx_free", counted)
    gc.collect()
    baseline = len(calls)
    for backend in ("python", "auto"):
        references = []
        for index in range(3):
            engine = Engine([Resource("r", 10.0)], backend=backend)
            engine.submit(f"pending{index}", {"r": 1.0},
                          metadata={"engine": engine})
            references.append(weakref.ref(engine))
            del engine
        gc.collect()
        assert all(reference() is None for reference in references), backend
    references = []
    for index in range(3):
        engine = Engine([Resource("r", 10.0)], backend="c")
        engine.submit(f"pending{index}", {"r": 1.0},
                      metadata={"engine": engine})
        references.append(weakref.ref(engine))
        del engine
    gc.collect()
    assert all(reference() is None for reference in references)
    assert len(calls) - baseline == 3

    engine = Engine([Resource("r", 10.0)], backend="c")
    engine.submit("pending", {"r": 1.0}, metadata={"engine": engine})
    engine._accel.close()
    assert len(calls) - baseline == 4
    del engine
    gc.collect()
    assert len(calls) - baseline == 4


def test_summary_fallback_reason_tracks_current_backend_state():
    """F7: summary() reports the same current-state reason as accel_status()."""
    for backend in ("python", "c", "auto"):
        engine = Engine([Resource("r", 1.0)], backend=backend)
        engine.submit("sleep", {}, latency=10.0)
        engine.advance(horizon=1.0)
        engine.submit("work", {"r": 1.0}, on_complete=lambda event: None)
        engine.advance()
        status = engine.accel_status()
        summary = engine.summary()
        assert summary["fallback_reason"] == status["fallback_reason"], backend
        assert summary["accel_backend"] == status["accel_backend"], backend
        if backend == "c":
            assert summary["fallback_reason"] == "mixed_frontier"
