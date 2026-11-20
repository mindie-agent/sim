"""Shared builders for the C11 ``_engine_accel`` parity and lifecycle tests.

[Role] Own the single definitions for the native-library memo, seeded random
  Engine programs sized from packaged Ascend geometry, the ULP metric, the
  finalizer/cache helpers, the one-shot allocation-fault wrapper and the
  PEP 517 wheel builder used across the parity and lifecycle modules.
[Invariants] Random programs are pure functions of ``(seed, sku)`` with every
  coupled placement obeying ``len(units_r)*res_0.units == len(units_0)*res_r.units``
  and staying within ``MAX_U``.  Wheel and fault-wrapper builds write only into
  test-owned temporary trees created through ``_support.tempdirs.tempdir``.
[Timeouts & Locks] Subprocess compiles and PEP 517 builds carry explicit
  timeouts; no sleeps or background threads.
[Side-Effect Policy] Writes only the session-isolated cache or test-owned
  temporary directories; never mutates the workspace or real user cache.
"""
from __future__ import annotations

import base64
import csv
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import subprocess
import sys
import time
import warnings
import weakref
import zipfile

from _support.tempdirs import tempdir
from mindie_sim.backends.ascend import profiles
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.runtime import accel_bridge, accel_build
from mindie_sim.prediction.runtime.accel_tables import Arena, install_signatures

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "mindie_sim" / "prediction" / "runtime"
NATIVE_DIR = ROOT / "tests" / "runtime" / "native"
BACKENDS = ("python", "c", "auto")
NATIVE = ("c", "auto")
#: 64-root capped frontier: root0 ends at 1.0, root1 at 2.0, the rest at 3.0.
FRONTIER_WORK = (1.0, 2.0) + (3.0,) * 62
PREBUILT_SUFFIXES = frozenset({".so", ".dylib", ".dll", ".o", ".a", ".pyc", ".pyo"})
BINARY_MAGIC = (
    b"\x7fELF", b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca", b"MZ", b"!<arch>\n",
)
_RATE_SCALE = 1e-12

_FAULT_WRAPPER = r"""
/* Test-only one-shot realloc failure shim around unchanged production C11. */
#include <stddef.h>
#include <stdlib.h>
static int drain_fault_armed;
static int drain_fault_hits;
static void *drain_fault_target;
static void *drain_fault_realloc(void *ptr, size_t bytes)
{
    if (drain_fault_armed && ptr == drain_fault_target) {
        drain_fault_armed = 0;
        drain_fault_hits++;
        return NULL;
    }
    return realloc(ptr, bytes);
}
#define realloc drain_fault_realloc
#include "_engine_accel.c"
#ifdef _WIN32
#define DRAIN_FAULT_EXP __declspec(dllexport)
#else
#define DRAIN_FAULT_EXP
#endif
#undef realloc

DRAIN_FAULT_EXP void drain_fault_arm(void *ctx, int kind)
{
    msim_ctx *c = (msim_ctx *)ctx;
    drain_fault_target = kind == 1 ? (void *)c->cheap
                       : kind == 2 ? (void *)c->heap : (void *)c->act;
    drain_fault_armed = 1;
}

DRAIN_FAULT_EXP int drain_fault_count(void) { return drain_fault_hits; }
"""


class Token:
    """A weakrefable custom leaf: never an inert built-in."""


def native_library():
    """The process-local production library (built once per session)."""
    status = accel_build.build_accel()
    assert status["ok"], status.get("detail") or status.get("reason")
    return accel_bridge.load_library(status)


def close(engine) -> None:
    if engine._accel is not None:
        engine._accel.close()


def uniform_engine(backend: str, count: int, units: int = 1, rate=None):
    engine = Engine([Resource("r", float(rate or count), units=units)],
                    backend=backend)
    return engine, [engine.submit(f"t{i}", {"r": 1.0}) for i in range(count)]


def ulps(left: float, right: float) -> int:
    """Distance in representable binary64 steps (sign-aware, finite)."""
    if left == right:
        return 0

    def ordered(value: float) -> int:
        raw = struct.unpack("<q", struct.pack("<d", value))[0]
        return raw if raw >= 0 else -(raw & 0x7FFFFFFFFFFFFFFF)

    return abs(ordered(left) - ordered(right))


def sku_resources(sku: str) -> list[Resource]:
    """Cube/vector/MTE/HBM resources with the SKU's declared owner counts."""
    device = profiles()[sku]["device"]
    units = device["resource_units"]
    rates = device["rates"]
    return [
        Resource("cube", device["compute_peaks"]["bf16"] * _RATE_SCALE,
                 units=device["compute_units"]),
        Resource("vector", rates["vector"] * _RATE_SCALE, units=units["vector"]),
        Resource("mte2", rates["mte2_aic"] * _RATE_SCALE,
                 units=units["mte2_aic"]),
        Resource("mte3", rates["mte3"] * _RATE_SCALE, policy="fifo"),
        Resource("hbm", rates["hbm"] * _RATE_SCALE),
    ]


def _proportional(rng: random.Random, resources) -> dict[str, tuple[int, ...]]:
    """A coupled cube+vector slice obeying the proportional placement law."""
    cube, vector = resources[0].units, resources[1].units
    ratio = vector // cube
    width = rng.randint(1, cube)
    start = rng.randrange(cube - width + 1)
    aic = tuple(range(start, start + width))
    aiv = tuple(range(start * ratio, (start + width) * ratio))
    return {"cube": aic, "vector": aiv}


def random_program(seed: int, sku: str, size: int = 96, kinds=range(6)):
    """A native-eligible random DAG: resources plus ``Engine.submit`` specs."""
    rng = random.Random(seed)
    resources = sku_resources(sku)
    kinds = tuple(kinds)
    tasks = []
    for index in range(size):
        kind = 2 if (index == 0 and size <= 24 and 2 in kinds) else rng.choice(kinds)
        spec: dict = {"after": tuple(sorted(rng.sample(
            range(index), k=min(index, rng.randrange(3)))))}
        amount = rng.uniform(0.25, 8.0)
        if kind == 0:
            spec["work"] = {"cube": amount * resources[0].rate}
        elif kind == 1:
            spec["work"] = {"vector": amount * resources[1].rate,
                            "mte2": rng.uniform(0.1, 2.0) * resources[2].rate}
            spec["caps"] = {"vector": resources[1].rate * rng.uniform(0.2, 1.0)}
        elif kind == 2:
            spec["work"] = {"cube": amount * resources[0].rate,
                            "vector": amount * resources[1].rate}
            spec["coupled"] = True
            spec["placement"] = _proportional(rng, resources)
        elif kind == 3:
            spec["work"] = {"mte3": amount * resources[3].rate,
                            "hbm": rng.uniform(0.1, 4.0) * resources[4].rate}
        elif kind == 4:
            units = resources[0].units
            spec["work"] = {"cube": amount}
            spec["placement"] = {"cube": tuple(sorted(rng.sample(
                range(units), k=rng.randint(1, units))))}
        else:
            spec["work"] = {"hbm": amount * resources[4].rate}
            spec["latency"] = rng.uniform(0.0, 0.5)
        if rng.random() < 0.15:
            spec["release"] = rng.uniform(0.0, 4.0)
        tasks.append(spec)
    return resources, tasks


def run_program(resources, tasks, backend: str, **engine_kwargs) -> dict:
    """Submit one spec list and observe every public clock/counter cell."""
    engine = Engine(resources, backend=backend, **engine_kwargs)
    events = []
    for index, spec in enumerate(tasks):
        kwargs = dict(spec)
        after = tuple(events[pred] for pred in kwargs.pop("after", ()))
        events.append(engine.submit(f"t{index}", kwargs.pop("work"),
                                    after=after, **kwargs))
    engine.advance()
    status = engine.accel_status()
    result = {
        "now": engine.now,
        "starts": [event.started for event in events],
        "times": [event.time for event in events],
        "busy": dict(engine.busy_ms),
        "service": dict(engine.serviced),
        "completed": engine.completed_count,
        "steps": status["native_steps"],
        "reason": status["fallback_reason"],
    }
    close(engine)
    return result


def max_ulps(left: dict, right: dict) -> int:
    """Largest binary64 distance across every numeric observation."""
    worst = ulps(left["now"], right["now"])
    for key in ("starts", "times"):
        for a, b in zip(left[key], right[key], strict=True):
            assert (a is None) == (b is None), key
            if a is not None:
                worst = max(worst, ulps(a, b))
    for key in ("busy", "service"):
        assert left[key].keys() == right[key].keys()
        for name, value in left[key].items():
            assert math.isfinite(value), (key, name)
            worst = max(worst, ulps(value, right[key][name]))
    return worst


# -- finalizer & cache lifecycle helpers -----------------------------------
def frontier_roots(engine, **first):
    """64-root capped frontier with optional overrides on root0."""
    roots = [engine.submit("root0", {"r": 1.0}, caps={"r": 1.0}, **first)]
    roots.extend(engine.submit(f"root{i}", {"r": amount}, caps={"r": 1.0})
                 for i, amount in enumerate(FRONTIER_WORK[1:], 1))
    return roots


def arm_observer(engine, token, target, follow, registration):
    """Register ``follow`` on ``target`` when ``token`` is finalized."""
    def register():
        registration.append((engine.now, target.done))
        engine.on_complete(target, follow)
    return weakref.finalize(token, register)


def make_counter_observer(engine, roots, target):
    """Return ``(register, registration, observed, extra)`` for finalizer tests."""
    registration, observed, extra = [], [], []

    def follow(event):
        observed.append((event.time, engine.now, engine.busy_ms["r"],
                         engine.serviced["r"]))
        extra.append(engine.submit("extra", {"r": engine.serviced["r"]},
                                   on_complete=lambda _event: None))

    def register():
        registration.append((engine.now, roots[target].done))
        engine.on_complete(roots[target], follow)

    return register, registration, observed, extra


def make_cache_artifact(cache: Path, index: int, extension: str, age: float):
    """Write one synthetic library + manifest pair aged ``age`` seconds."""
    digest = f"{index:064x}"
    library = cache / f"{digest}-0{extension}"
    library.write_bytes(b"native")
    manifest = cache / (library.name + ".json")
    manifest.write_text(json.dumps({"digest": digest, "library": str(library)}))
    stamp = time.time() - age
    os.utime(library, (stamp, stamp))
    os.utime(manifest, (stamp, stamp))
    return library, manifest


def compile_fault_library():
    """Compile unchanged ``_engine_accel.c`` with a one-shot failing ``realloc``."""
    compiler, _ = accel_build.discover_compiler()
    assert compiler, "a C compiler is required for drain-fault probes"
    directory = tempdir("accel-drain-fault-")
    source = directory / "fault_wrapper.c"
    source.write_text(_FAULT_WRAPPER, encoding="ascii")
    output = directory / f"fault_wrapper{accel_build.library_extension()}"
    pic = () if os.name == "nt" else ("-fPIC",)
    command = [compiler, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror",
               "-ffp-contract=off", *pic, "-shared", "-I", str(RUNTIME),
               str(source), "-o", str(output)]
    done = subprocess.run(command, capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr
    library = ctypes.CDLL(str(output))
    install_signatures(library)
    library.drain_fault_arm.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.drain_fault_arm.restype = None
    library.drain_fault_count.argtypes = []
    library.drain_fault_count.restype = ctypes.c_int
    return library, directory


# -- PEP 517 wheel helpers -------------------------------------------------
def package_files() -> list[Path]:
    """Every tracked non-cache file in the workspace ``mindie_sim`` tree."""
    return sorted(
        path for path in (ROOT / "mindie_sim").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"})


def wheel_env(base: Path) -> dict[str, str]:
    env = dict(os.environ)
    for name in ("MINDIE_SIM_ACCEL", "MINDIE_SIM_ENGINE_BACKEND",
                 "MINDIE_SIM_CC", "MINDIE_DATA_HOME", "XDG_CACHE_HOME",
                 "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
               TMPDIR=str(base), MINDIE_CACHE_HOME=str(base / "build-cache"))
    return env


def verify_wheel_record(wheel: Path) -> dict[str, bytes]:
    """Verify RECORD hashes (via distlib when present, else stdlib) and return entries."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            from distlib.wheel import Wheel
        except ModuleNotFoundError:
            Wheel = None
    if Wheel is not None:
        Wheel(str(wheel)).verify()
    with zipfile.ZipFile(wheel) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()
                   if not name.endswith("/")}
    if Wheel is None:
        rec_name = next(n for n in entries if n.endswith(".dist-info/RECORD"))
        rows = list(csv.reader(entries[rec_name].decode("utf-8").splitlines()))
        assert {row[0] for row in rows} == set(entries)
        for name, digest, size in rows:
            if name == rec_name:
                assert digest == "" and size == ""
                continue
            algo, expected = digest.split("=", 1)
            actual = base64.urlsafe_b64encode(
                hashlib.new(algo, entries[name]).digest()
            ).rstrip(b"=").decode("ascii")
            assert actual == expected and int(size) == len(entries[name]), name
    return entries


def build_pep517_wheel(base: Path) -> Path:
    """Build a real PEP 517 universal wheel from an isolated copy under ``base``."""
    source = base / "source"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(ROOT / "mindie_sim", source / "mindie_sim", ignore=ignore)
    shutil.copy2(ROOT / "pyproject.toml", source / "pyproject.toml")
    readme = ROOT / "README.md"
    if readme.exists():
        shutil.copy2(readme, source / "README.md")
    (source / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    wheel_dir = base / "dist"
    wheel_dir.mkdir()
    hook = base / "pep517_hook.py"
    hook.write_text(
        "import sys\nfrom pathlib import Path\n"
        "from setuptools import build_meta\n"
        "print(build_meta.build_wheel(str(Path(sys.argv[1]))))\n")
    attempts = []
    result = subprocess.run(
        [sys.executable, "-B", str(hook), str(wheel_dir)],
        cwd=source, env=wheel_env(base), capture_output=True, text=True,
        timeout=300)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if result.returncode == 0 and wheels:
        return wheels[0]
    attempts.append(("setuptools.build_meta.build_wheel", result))
    for path in wheel_dir.glob("*.whl"):
        path.unlink()
    for name in ("build", "mindie_sim.egg-info"):
        shutil.rmtree(source / name, ignore_errors=True)
    result = subprocess.run(
        [sys.executable, "-B", "-m", "build", "--wheel", "--outdir", str(wheel_dir), "."],
        cwd=source, env=wheel_env(base), capture_output=True, text=True, timeout=600)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if result.returncode == 0 and wheels:
        return wheels[0]
    attempts.append(("python -m build --wheel", result))
    detail = "\n\n".join(f"[{n}] exit={p.returncode}\n{p.stdout}\n{p.stderr}" for n, p in attempts)
    raise AssertionError(f"real PEP 517 wheel build failed:\n{detail}")


def install_and_smoke_wheel(base: Path, wheel: Path) -> tuple[Path, dict]:
    """Install ``wheel`` into an isolated target tree and run the smoke probe."""
    extract, run_dir = base / "installed", base / "run"
    run_dir.mkdir()
    inst = subprocess.run(
        [sys.executable, "-B", "-m", "pip", "install", "--no-deps", "--no-index",
         "--no-compile", "--no-cache-dir", "--target", str(extract), str(wheel)],
        env=wheel_env(base), capture_output=True, text=True, timeout=240)
    assert inst.returncode == 0, (inst.stdout, inst.stderr)
    script = base / "smoke.py"
    script.write_text(WHEEL_SMOKE_SCRIPT, encoding="utf-8")
    env = {**wheel_env(base), "PYTHONPATH": str(extract), "MINDIE_CACHE_HOME": str(base / "cache")}
    done = subprocess.run([sys.executable, "-B", str(script), str(extract)],
                          cwd=run_dir, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, (done.stdout, done.stderr)
    return extract, json.loads(done.stdout.strip().splitlines()[-1])


def build_and_run_sanitizer(tmp_path: Path) -> tuple[subprocess.CompletedProcess, list[subprocess.CompletedProcess]]:
    """Compile and run the C11 ASan/UBSan harness plus strict ``-Werror`` object builds."""
    source, harness = RUNTIME / "_engine_accel.c", NATIVE_DIR / "accel_sanitizer_harness.c"
    compiler, reason = accel_build.discover_compiler()
    assert compiler is not None, reason or "compiler_unavailable"
    cc = str(Path(compiler).resolve()) if os.name == "nt" else (shutil.which("clang") or str(Path(compiler).resolve()))
    binary = tmp_path / ("accel_sanitizer.exe" if os.name == "nt" else "accel_sanitizer")
    san = () if os.name == "nt" else ("-fsanitize=address,undefined", "-fno-sanitize-recover=all")
    libm = () if os.name == "nt" else ("-lm", "-pthread")
    base = [cc, "-std=c11", "-O1", "-g", "-ffp-contract=off", "-I", str(RUNTIME),
            str(source), str(harness), "-o", str(binary), *libm]
    built = subprocess.run([*base[:4], *san, *base[4:]], capture_output=True, text=True, timeout=240)
    if built.returncode != 0 and san:
        built = subprocess.run(base, capture_output=True, text=True, timeout=240)
    assert built.returncode == 0, built.stderr[-2000:]
    env = {**os.environ, "ASAN_OPTIONS": "detect_leaks=0:halt_on_error=1",
           "UBSAN_OPTIONS": "halt_on_error=1:print_stacktrace=1"}
    ran = subprocess.run([str(binary)], capture_output=True, text=True, timeout=240, env=env)
    strict = [subprocess.run(
        [cc, "-std=c11", "-O1", "-Wall", "-Wextra", "-Werror", "-ffp-contract=off",
         "-I", str(RUNTIME), "-c", str(src), "-o", str(tmp_path / obj)],
        capture_output=True, text=True, timeout=180)
        for src, obj in ((source, "accel.o"), (harness, "harness.o"))]
    return ran, strict


# -- boundary & raw-ABI builders -------------------------------------------
def open_arena(rates=(10.0,)) -> Arena:
    arena = Arena(native_library())
    arena.configure(list(rates), [1] * len(rates), [0] * len(rates))
    return arena


def arena_row(seq=1, prio=0, cohort=1, waiting=0, flags=0, res=0):
    return (seq, flags, prio, cohort, waiting, 0.0, 0.0, 0.0,
            [(res, 1.0, math.inf, ())])


def _mixed_resources(backend):
    engine = Engine([Resource("mte", 64.0), Resource("cube", 128.0, units=4),
                     Resource("vec", 32.0, units=2, policy="fifo")], backend=backend)
    events = [engine.submit(f"t{i}", {"mte": 2.0 + i % 5, "cube": 4.0 + i % 3,
                                      "vec": 1.0 + i % 2}) for i in range(48)]
    engine.advance()
    return engine, (engine.now, [e.time for e in events], dict(engine.busy_ms))


def _fifo_and_trace(backend):
    engine = Engine([Resource("f", 1.0, policy="fifo")], backend=backend, trace_limit=16)
    events = [engine.submit("w1", {"f": 1.0}), engine.submit("w2", {"f": 2.0})]
    engine.advance()
    return engine, ([e.time for e in events],
                    [(row["task"], row["end_ms"]) for row in engine.trace])


def _incremental_placement(backend):
    engine = Engine([Resource("r", 2.0, units=2)], backend=backend)
    first = engine.submit("A", {"r": 2.0}, placement={"r": (0,)})
    engine.advance(horizon=0.5)
    second = engine.submit("B", {"r": 1.0}, placement={"r": (1,)})
    engine.advance()
    return engine, (first.time, second.time, dict(engine.serviced), engine.now)


def _placement_burst(count, units, coupled=False, last=None):
    def build(backend):
        engine = Engine([Resource("r", float(units), units=units)], backend=backend)
        events = [engine.submit(f"t{i}", {"r": 1.0}, coupled=coupled,
                                placement={"r": tuple(range(
                                    last if last and i == count - 1 else units))})
                  for i in range(count)]
        engine.advance()
        return engine, (engine.now, events[0].time, events[-1].time, dict(engine.serviced))
    return build


def _sequential_placements(backend):
    engine = Engine([Resource("r", 64.0, units=64)], backend=backend)
    if backend != "c":
        for i in range(63):
            engine.submit(f"w{i}", {})
        engine.advance()
    units, times = tuple(range(64)), []
    for i in range(1100):
        event = engine.submit(f"t{i}", {"r": 1.0}, placement={"r": units})
        engine.advance(until=(event,))
        times.append(event.time)
    return engine, (engine.now, times[0], times[-1])


def _delayed_root(warm_work, warm_count, horizons, rate=64.0):
    def build(backend):
        engine = Engine([Resource("r", rate)], backend=backend)
        warm = [engine.submit(f"w{i}", dict(warm_work)) for i in range(warm_count)]
        root = engine.submit("A", {"r": 1.0}, release=2.0)
        follower = engine.submit("B", {"r": 1.0}, after=(root,))
        seen = []
        for horizon in horizons:
            engine.advance(horizon=horizon)
            seen.append((engine.now, root.started, follower.started))
        engine.advance()
        assert engine._releases == []
        return engine, (seen, engine.now, warm[-1].time, root.started,
                        root.time, follower.time, engine.completed_count)
    return build


def _start_burst(backend):
    engine, events = uniform_engine(backend, 4097)
    engine.advance()
    return engine, (engine.now, all(e.started == 0.0 and e.done for e in events))


BOUNDARY_CASES = (
    ("mixed-fair-fifo-units", _mixed_resources, False),
    ("fifo-order-and-trace-end", _fifo_and_trace, False),
    ("incremental-placement", _incremental_placement, False),
    ("coupled-equal-placements-interned", _placement_burst(257, 64, coupled=True), True),
    ("overlapping-unit-membership", _placement_burst(1025, 64, last=63), True),
    ("simultaneous-placement-pool-growth", _placement_burst(2000, 64), True),
    ("sequential-placement-pool-reuse", _sequential_placements, True),
    ("delayed-release-adoption", _delayed_root({}, 62, (), rate=1.0), True),
    ("delayed-release-horizon-continuation", _delayed_root({"r": 1.0}, 62, (1.0, 2.5)), True),
    ("native-delayed-release", _delayed_root({"r": 1.0}, 64, ()), True),
    ("start-buffer-continuation", _start_burst, True),
)


WHEEL_SMOKE_SCRIPT = r'''
import json, sys, weakref
from pathlib import Path
import mindie_sim
from mindie_sim.backends.ascend import profiles as ascend_profiles
from mindie_sim.backends.nvidia import profiles as nvidia_profiles
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.runtime.accel_tables import metadata_is_stable

extract = Path(sys.argv[1]).resolve()
package = Path(mindie_sim.__file__).resolve()
assert package.is_relative_to(extract), (package, extract)
assert len(ascend_profiles()) == 16 and len(nvidia_profiles()) == 4
assert type(load_model("qwen3-32b")).__name__ == "ModelSpec"
class F(float): pass
class Tok: pass
report = {"package": str(package)}
for mode in ("c", "auto"):
    e = Engine([Resource("r", 4097.0)], backend=mode, trace_limit=3)
    first = e.submit("root0", {"r": 1.0})
    child = e.submit("child", {}, after=(first,))
    roots = [first] + [e.submit(f"root{i}", {"r": 1.0}) for i in range(1, 4097)]
    e.advance(horizon=1.0)
    assert e.now == 1.0 and e.completed_count == 4098 and child.time == 1.0
    assert [row["task"] for row in e.trace] == ["root4095", "root4096", "child"]
    st = e.accel_status()
    assert st["native_steps"] == 1 and st["accel_backend"] == "c_native"
    report[mode] = st["native_steps"]
    e._accel.close()
for mode in ("python", "c", "auto"):
    e = Engine([Resource("r", 64.0)], backend=mode, trace_limit=0)
    tok = Tok()
    r0 = e.submit("root0", {"r": 1.0}, caps={"r": 1.0}, metadata={"token": tok})
    ch = e.submit("child", {"r": 1.0}, caps={"r": 1.0}, after=(r0,))
    for i, a in enumerate([2.0] + [3.0] * 62, 1):
        e.submit(f"root{i}", {"r": a}, caps={"r": 1.0})
    seen, extra = [], []
    def rel():
        seen.append((e.now, r0.done, ch.started, ch.done, e.completed_count, e.serviced["r"]))
        extra.append(e.submit("extra", {"r": 1.0 if ch.started is None else 2.0}))
    weakref.finalize(tok, rel); del tok
    e.advance()
    report[f"desc_{mode}"] = [seen, [(x.started, x.time) for x in extra], ch.started, ch.time,
                              e.now, e.serviced["r"], e.completed_count, e.accel_status()["native_steps"]]
    if e._accel is not None: e._accel.close()
nested = []
for _ in range(9): nested = (nested,)
report["stable"] = metadata_is_stable({"payload": nested})
print(json.dumps(report))
'''
