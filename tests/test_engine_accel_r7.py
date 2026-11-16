"""Round-7 regressions: drain-failure publication and snapshot continuation.

[Role] Reproduce the two R7 counterexamples against the real kernel: an
  allocation failure inside the zero-budget output drain must publish its
  accepted prefix (and prior counters), and an output continuation must finish
  the interrupted completion snapshot before any awakened successor starts.
[Invariants] Faults are real ``MSIM_E_RANGE`` exits produced by an allocation
  shim that includes the unchanged production source; no Python
  ``KernelFault`` is fabricated.  ``Engine._rates`` is forbidden during every
  failing advance, establishing that no Python replay happens.
[Timeouts & Locks] No threads or sleeps; compilation and every probe carry
  explicit subprocess/iteration bounds.
[Side-Effect Policy] Compiles only into a temporary directory; never writes
  the package cache or the workspace.
"""
from __future__ import annotations

import ctypes
from pathlib import Path
import shutil
import subprocess
from unittest.mock import patch

import pytest

from temp_cleanup import tempdir

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "mindie_sim" / "prediction" / "runtime"

from mindie_sim.prediction.engine import Engine, Resource  # noqa: E402
from mindie_sim.prediction.runtime import (  # noqa: E402
    accel_bridge, accel_build,
)
from mindie_sim.prediction.runtime.accel_tables import (  # noqa: E402
    Arena, KernelFault, install_signatures,
)

BACKENDS = ("python", "c", "auto")
WRAPPER = r"""
/* R7-1 test-only allocation-failure shim; includes unchanged production code. */
#include <stddef.h>
#include <stdlib.h>
static int r7_armed;
static int r7_hits;
static void *r7_target;
static void *r7_realloc(void *ptr, size_t bytes)
{
    if (r7_armed && ptr == r7_target) {
        r7_armed = 0;
        r7_hits++;
        return NULL;
    }
    return realloc(ptr, bytes);
}
#define realloc r7_realloc
#include "_engine_accel.c"
#ifdef _WIN32
#define R7_EXP __declspec(dllexport)
#else
#define R7_EXP
#endif
#undef realloc

R7_EXP void r7_arm(void *ctx, int kind)
{
    msim_ctx *c = (msim_ctx *)ctx;
    r7_target = kind == 1 ? (void *)c->cheap
              : kind == 2 ? (void *)c->heap : (void *)c->act;
    r7_armed = 1;
}

R7_EXP int r7_failures(void) { return r7_hits; }
"""


@pytest.fixture(scope="module")
def fault_library():
    """Compile the unchanged kernel with a one-shot failing ``realloc``."""
    compiler, _ = accel_build.discover_compiler()
    assert compiler, "a C compiler is required for the R7 fault probes"
    directory = tempdir("nr-s3-r7-fault-")
    source = Path(directory) / "fault_wrapper.c"
    source.write_text(WRAPPER, encoding="ascii")
    output = Path(directory) / f"fault_wrapper{accel_build.library_extension()}"
    pic = () if __import__("os").name == "nt" else ("-fPIC",)
    command = [compiler, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
               "-ffp-contract=off", *pic, "-shared", "-I", str(RUNTIME),
               str(source), "-o", str(output)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr
    library = ctypes.CDLL(str(output))
    install_signatures(library)
    library.r7_arm.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.r7_arm.restype = None
    library.r7_failures.argtypes = []
    library.r7_failures.restype = ctypes.c_int
    try:
        yield library
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def _probe(library, backend, scenario):
    """Run one genuine drain-allocation failure and return the observed state."""
    original = Arena.run
    armed = [False]

    def run(self, now, horizon, next_release, budget):
        if budget == 0 and not armed[0] and (
                scenario == "admission" or now == 1.0):
            kind = {"admission": 1, "completion": 2, "no-prefix": 3}[scenario]
            library.r7_arm(self.ctx, kind)
            armed[0] = True
        return original(self, now, horizon, next_release, budget)

    result = {"scenario": scenario}
    with patch.object(accel_bridge, "load_library", lambda _status: library), \
            patch.object(accel_build, "build_accel",
                         lambda **_kw: {"ok": True, "cache_path": "r7-fault"}), \
            patch.object(Arena, "run", run):
        rate = 1.0 if scenario == "no-prefix" else 4097.0
        engine = Engine([Resource("r", rate)], backend=backend)
        before = library.r7_failures()
        if scenario == "admission":
            events = [engine.submit(f"z{i}", {}) for i in range(4097)]
            failing = lambda: engine.advance(horizon=0.0)
        elif scenario == "completion":
            events = [engine.submit(f"root{i}", {"r": 1.0})
                      for i in range(4097)]
            for i in range(8193):
                engine.submit(f"child{i}", {"r": 1.0}, after=(events[-1],))
            engine.advance(horizon=0.0)
            failing = lambda: engine.advance(horizon=1.0)
        else:
            good = engine.submit("good", {"r": 0.5})
            events = [engine.submit(f"late{i}", {"r": 1.0}, release=1.0)
                      for i in range(4097)]
            engine.advance(horizon=0.0)
            failing = lambda: engine.advance(horizon=1.0)
            result["good"] = good
        with patch.object(Engine, "_rates",
                          side_effect=AssertionError("Python replay forbidden")):
            with pytest.raises(KernelFault) as info:
                failing()
        assert library.r7_failures() == before + 1
        result.update(error=info.value, events=events, engine=engine)
    return result


@pytest.mark.parametrize("backend", ("c", "auto"))
def test_r7_1_admission_prefix_survives_a_drain_failure(fault_library, backend):
    """The final start accepted by the failing drain is published."""
    probe = _probe(fault_library, backend, "admission")
    engine, events, error = probe["engine"], probe["events"], probe["error"]
    assert error.has_prefix and error.prefix == (0.0, 0, 0, 0, 1)
    assert error.code == 1
    assert all(event.started == 0.0 for event in events)
    assert engine.now == 0.0 and engine.advance_steps == 0
    assert engine.completed_count == 0
    assert engine.accel_status()["native_calls"] == 2


@pytest.mark.parametrize("backend", ("c", "auto"))
def test_r7_1_completion_prefix_and_counters_survive_a_drain_failure(
        fault_library, backend):
    """The final completion and the preceding call's counters are published."""
    probe = _probe(fault_library, backend, "completion")
    engine, events, error = probe["engine"], probe["events"], probe["error"]
    assert error.has_prefix and error.prefix == (1.0, 0, 0, 1, 0)
    assert engine.now == 1.0 and engine.advance_steps == 1
    assert engine.completed_count == 4097
    assert events[-1].time == 1.0
    assert engine.busy_ms["r"] == 1.0
    assert engine.serviced["r"] == 4097.0


@pytest.mark.parametrize("backend", ("c", "auto"))
def test_r7_1_no_prefix_failure_still_publishes_prior_counters(
        fault_library, backend):
    """A failing drain that accepts nothing keeps the prior call's counters."""
    probe = _probe(fault_library, backend, "no-prefix")
    engine, events, error = probe["engine"], probe["events"], probe["error"]
    assert not error.has_prefix and error.prefix is None
    assert error.code == 1
    assert probe["good"].time == 0.5
    assert engine.now == 1.0 and engine.advance_steps == 2
    assert sum(event.started is not None for event in events) == 4096
    assert engine.busy_ms["r"] == 0.5
    assert engine.serviced["r"] == 0.5


def _completion_snapshot(backend, limit):
    engine = Engine([Resource("r", 4097.0)], backend=backend,
                    trace_limit=limit)
    roots = [engine.submit(f"root{i}", {"r": 1.0}) for i in range(4096)]
    engine.submit("child", {}, after=(roots[0],))
    engine.submit("last-root", {"r": 1.0})
    engine.advance(horizon=1.0)
    return engine


def _service_snapshot(backend, limit):
    names = [f"r{i}" for i in range(32)]
    work = {name: 1.0 for name in names}
    engine = Engine([Resource(name, 1025.0) for name in names],
                    backend=backend, trace_limit=limit)
    roots = [engine.submit(f"root{i}", work) for i in range(1025)]
    engine.submit("child", {}, after=(roots[0],))
    engine.advance(horizon=1.0)
    return engine


def _names(engine):
    return [row["task"] for row in engine.trace]


@pytest.mark.parametrize("backend", BACKENDS)
def test_r7_2_completion_buffer_continuation_keeps_snapshot_order(backend):
    """4,096 completions fill the wire buffer; ``last-root`` still precedes child."""
    full = _completion_snapshot(backend, 8192)
    assert _names(full) == [f"root{i}" for i in range(4096)] + [
        "last-root", "child"]
    assert full.now == 1.0 and full.completed_count == 4098
    assert full.trace_dropped == 0
    assert _names(_completion_snapshot(backend, 1)) == ["child"]
    assert _names(_completion_snapshot(backend, 3)) == [
        "root4095", "last-root", "child"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_r7_2_service_buffer_continuation_keeps_snapshot_order(backend):
    """1,024 x 32 records fill the service buffer; the last root precedes child."""
    full = _service_snapshot(backend, 2048)
    assert _names(full) == [f"root{i}" for i in range(1025)] + ["child"]
    assert full.now == 1.0 and full.completed_count == 1026
    assert full.trace_dropped == 0
    assert all(value == 1025.0 for value in full.serviced.values())
    assert _names(_service_snapshot(backend, 1)) == ["child"]
    assert _names(_service_snapshot(backend, 3)) == [
        "root1023", "root1024", "child"]


def test_r7_modules_obey_headers_and_encoding():
    tags = ("[Role]", "[Invariants]", "[Timeouts & Locks]",
            "[Side-Effect Policy]")
    for path in (RUNTIME / "accel_sync.py", RUNTIME / "accel_bridge.py",
                 RUNTIME / "_engine_accel_core.h",
                 RUNTIME / "_engine_accel_abi.h",
                 Path(__file__).with_name("accel_sanitizer_r7.h"),
                 Path(__file__).with_name("accel_sanitizer_harness.c"),
                 Path(__file__)):
        data = path.read_bytes()
        assert b"\r" not in data and data.endswith(b"\n"), path.name
        assert len(data.splitlines()) < 745, path.name
        head = "\n".join(data.decode().splitlines()[:30])
        assert all(tag in head for tag in tags), path.name
