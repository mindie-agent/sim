"""Round-9 closure regressions: hand-back, causal callbacks and the wheel.

[Role] Reproduce an early-batch decline during ``_apply`` (wide priority and
  capacity), a finalizer-registered observer across multi-instant native
  integration, and the installed-wheel runtime-resource contract.
[Invariants] Every fixture uses the unchanged production library and public
  Engine APIs.  Registration is a deterministic ``weakref.finalize`` trigger;
  no fabricated ``KernelFault``, allocator change or second scheduler exists.
  The wheel is built and extracted only under a test-owned temporary tree; the
  workspace package files are read-only inputs compared byte-for-byte.
[Timeouts & Locks] No threads or sleeps; native calls stay inside the bounded
  per-epoch budget, the isolated build memo and the subprocess timeout.
[Side-Effect Policy] Compiles only into the isolated temporary cache and writes
  only test-owned temporary trees; never writes the workspace or user cache.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings
import weakref
import zipfile

import pytest

from temp_cleanup import tempdir
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.runtime.accel_tables import _is_inert_builtin

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "mindie_sim" / "prediction" / "runtime"
BACKENDS = ("python", "c", "auto")
NATIVE = ("c", "auto")
COUNTS = (64, 4097)
CAPACITY_EDGES = 32769
PACKAGE_FILES = 504
TOTAL_ENTRIES = 508
DIST_INFO = frozenset({
    "mindie_sim-0.1.0.dist-info/METADATA",
    "mindie_sim-0.1.0.dist-info/WHEEL",
    "mindie_sim-0.1.0.dist-info/top_level.txt",
    "mindie_sim-0.1.0.dist-info/RECORD",
})
PREBUILT_SUFFIXES = {".so", ".dylib", ".dll", ".o", ".a", ".pyc", ".pyo"}
BINARY_MAGIC = (
    b"\x7fELF", b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe",
    b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe", b"\xca\xfe\xba\xbe",
    b"\xbe\xba\xfe\xca", b"MZ", b"!<arch>\n",
)


class _Token:
    """Metadata payload whose release deterministically arms an observer."""


def _arm(engine, token, target, follow, registration):
    """Register ``follow`` on ``target`` when ``token`` is released."""
    def register():
        registration.append((engine.now, target.done))
        engine.on_complete(target, follow)
    return weakref.finalize(token, register)


def _close(engine) -> None:
    if engine._accel is not None:
        engine._accel.close()


def _assert_same(left, right, label):
    assert left == right, (label, left, right)


def _wide_priority(backend, count):
    """A legal wide priority submitted by a callback on the second root."""
    engine = Engine([Resource("r", float(count))], backend=backend,
                    trace_limit=0)
    token = _Token()
    first = engine.submit("root0", {"r": 1.0}, metadata={"token": token})
    roots = [first] + [engine.submit(f"root{index}", {"r": 1.0})
                       for index in range(1, count)]
    registration, observed, extra = [], [], []

    def follow(event):
        observed.append((event.time, engine.now, engine.busy_ms["r"],
                         engine.serviced["r"]))
        extra.append(engine.submit("extra", {"r": 1.0}, priority=2 ** 63))

    finalizer = _arm(engine, token, roots[1], follow, registration)
    del token
    engine.advance()
    assert not finalizer.alive
    status = engine.accel_status()
    result = {
        "registration": list(registration),
        "observed": list(observed),
        "extra": [(event.started, event.time) for event in extra],
        "now": engine.now,
        "busy": engine.busy_ms["r"],
        "service": engine.serviced["r"],
        "completed": engine.completed_count,
        "roots": sorted({event.time for event in roots}),
        "steps": status["native_steps"],
        "reason": status["backend_reason"],
        "pending": len(engine._accel.pending) if engine._accel else 0,
        "native_seqs": len(engine._native_seqs),
        "native_slots": len(engine._native_slots),
    }
    _close(engine)
    return result


@pytest.mark.parametrize("count", COUNTS)
def test_r9_1_wide_priority_decline_finishes_publication_exactly_once(count):
    """A decline during ``_apply`` never re-executes an accepted completion."""
    results = {backend: _wide_priority(backend, count) for backend in BACKENDS}
    control = results["python"]
    assert control["registration"] == [(1.0, False)]
    assert control["observed"] == [(1.0, 1.0, 1.0, float(count))]
    assert control["roots"] == [1.0]
    assert control["extra"][0][0] == 1.0
    assert control["service"] == float(count + 1)
    assert control["completed"] == count + 1
    assert control["now"] == pytest.approx(1.0 + 1.0 / count, abs=1e-12)
    for backend in NATIVE:
        result = results[backend]
        # R12-2: the token root declines native.  ``c`` attaches early and
        # reports the wide-integer decline; ``auto`` meets its threshold on the
        # mixed frontier first.  Both match Python exactly.
        assert result["steps"] == 0, backend
        assert result["reason"] == (
            "wide_integer" if backend == "c" else "auto_mixed_frontier")
        for key in ("registration", "observed", "extra", "now", "busy",
                    "service", "completed", "roots"):
            _assert_same(result[key], control[key], (backend, key))
        assert (result["pending"], result["native_seqs"],
                result["native_slots"]) == (0, 0, 0)


def _capacity_decline(backend, count):
    """A callback submits a row whose dependency frontier exceeds MAX_OWN."""
    engine = Engine([Resource("r", float(count))], backend=backend,
                    trace_limit=0)
    token = _Token()
    first = engine.submit("root0", {"r": 1.0}, metadata={"token": token})
    roots = [first] + [engine.submit(f"root{index}", {"r": 1.0})
                       for index in range(1, count)]
    registration, observed, extra = [], [], []

    def follow(event):
        observed.append((event.time, engine.now, engine.busy_ms["r"],
                         engine.serviced["r"]))
        waits = [engine.future(f"wait{index}")
                 for index in range(CAPACITY_EDGES)]
        task = engine.submit("extra", {"r": 1.0}, after=tuple(waits))
        extra.append(task)
        for wait in waits:
            engine.resolve(wait)

    finalizer = _arm(engine, token, roots[1], follow, registration)
    del token
    engine.advance()
    assert not finalizer.alive
    status = engine.accel_status()
    result = {
        "registration": list(registration),
        "observed": list(observed),
        "extra": [(event.started, event.time) for event in extra],
        "now": engine.now,
        "busy": engine.busy_ms["r"],
        "service": engine.serviced["r"],
        "completed": engine.completed_count,
        "roots": sorted({event.time for event in roots}),
        "steps": status["native_steps"],
        "reason": status["backend_reason"],
        "pending": len(engine._accel.pending) if engine._accel else 0,
        "native_seqs": len(engine._native_seqs),
        "native_slots": len(engine._native_slots),
    }
    _close(engine)
    return result


@pytest.mark.parametrize("count", COUNTS)
def test_r9_1_capacity_decline_finishes_publication_exactly_once(count):
    """The capacity decline entry point shares the deferred hand-back path."""
    results = {backend: _capacity_decline(backend, count)
               for backend in BACKENDS}
    control = results["python"]
    assert control["registration"] == [(1.0, False)]
    assert control["observed"] == [(1.0, 1.0, 1.0, float(count))]
    assert control["roots"] == [1.0]
    assert control["extra"][0][0] == 1.0
    assert control["service"] == float(count + 1)
    assert control["completed"] == count + 1
    for backend in NATIVE:
        result = results[backend]
        # R12-2: ``c`` attaches early and reports the capacity decline;
        # ``auto`` meets its threshold on the mixed frontier first.
        assert result["steps"] == 0, backend
        assert result["reason"] == (
            "capacity_exceeded" if backend == "c" else "auto_mixed_frontier")
        for key in ("registration", "observed", "extra", "now", "busy",
                    "service", "completed", "roots"):
            _assert_same(result[key], control[key], (backend, key))
        assert (result["pending"], result["native_seqs"],
                result["native_slots"]) == (0, 0, 0)


def test_r9_2_inert_builtin_is_exact_type_strict():
    """Exact built-ins only: subclasses and custom payloads are never inert."""
    class MyInt(int):
        pass

    class MyStr(str):
        pass

    assert _is_inert_builtin(None)
    assert _is_inert_builtin((True, 2.5, "x", b"y", (1, "z")))
    assert _is_inert_builtin(())
    # R13-1: an exact frozenset is weakrefable, so it is never inert.
    assert not _is_inert_builtin(frozenset({1, 2.5}))
    assert not _is_inert_builtin((1, frozenset({2})))
    assert not _is_inert_builtin(MyInt(1))
    assert not _is_inert_builtin(MyStr("x"))
    assert not _is_inert_builtin({"token": _Token()})
    assert not _is_inert_builtin({"a": [1, 2.5, (True, "x", b"y")]})
    assert not _is_inert_builtin([[[[[1]]]]])
    assert not _is_inert_builtin([1, [2, [3, [4, [5, [6, [7, [8, [9]]]]]]]]])


def _causal(backend, case, kind):
    """A finalizer-registered observer across one native multi-instant batch."""
    engine = Engine([Resource("r", 64.0)], backend=backend, trace_limit=0)
    token = _Token()
    if case == "wide":
        work, caps = [1.0] * 64, None
    else:
        work, caps = [1.0, 2.0] + [3.0] * 62, {"r": 1.0}
    if case == "counter-uncapped":
        caps = None
    roots = [engine.submit(f"root{index}", {"r": amount}, caps=caps,
                           metadata={"token": token if index == 0 else None})
             for index, amount in enumerate(work)]
    registration, observed, extra = [], [], []

    def follow(event):
        observed.append((event.time, engine.now, engine.busy_ms["r"],
                         engine.serviced["r"]))
        if kind == "derived":
            extra.append(engine.submit(
                "extra", {"r": engine.serviced["r"]},
                on_complete=lambda _event: None))

    finalizer = _arm(engine, token, roots[1], follow, registration)
    del token
    engine.advance()
    assert not finalizer.alive
    status = engine.accel_status()
    result = {
        "registration": list(registration),
        "observed": list(observed),
        "extra": [(event.started, event.time) for event in extra],
        "now": engine.now,
        "busy": engine.busy_ms["r"],
        "service": engine.serviced["r"],
        "completed": engine.completed_count,
        "roots": [(event.started, event.time) for event in roots],
        "steps": status["native_steps"],
        "reason": status["backend_reason"],
    }
    _close(engine)
    return result


@pytest.mark.parametrize("case", ("counter-capped", "counter-uncapped", "wide"))
@pytest.mark.parametrize("kind", ("observe", "derived"))
def test_r9_2_callback_state_is_causal_across_native_instants(case, kind):
    """Registration and callback state match Python bit-for-bit, never future."""
    results = {backend: _causal(backend, case, kind) for backend in BACKENDS}
    control = results["python"]
    assert control["registration"] == [(1.0, False)]
    if case == "counter-capped":
        expected = (2.0, 2.0, 2.0, 127.0)
        assert control["observed"] == [expected]
        extra_at, extra_done = 2.0, 4.953125
    elif case == "counter-uncapped":
        expected = (1.984375, 1.984375, 1.984375, 127.0)
        assert control["observed"] == [expected]
        extra_at, extra_done = 1.984375, 4.9375
    else:
        expected = (1.0, 1.0, 1.0, 64.0)
        assert control["observed"] == [expected]
        extra_at, extra_done = 1.0, 2.0
    if kind == "derived":
        assert control["extra"] == [(extra_at, extra_done)]
        expected_service = 128.0 if case == "wide" else 316.0
        assert control["service"] == expected_service
        assert control["completed"] == 65
    else:
        assert control["extra"] == []
    for backend in NATIVE:
        result = results[backend]
        # R12-2: finalizer-bearing metadata declines native before integration.
        assert result["steps"] == 0, backend
        assert result["reason"] in {
            "mixed_frontier", "auto_mixed_frontier", "no_native_window",
            "auto_deferred"}, result["reason"]
        for key in ("registration", "observed", "extra", "now", "busy",
                    "service", "completed", "roots"):
            _assert_same(result[key], control[key], (backend, key))


def _package_files():
    """Every tracked non-cache file in the workspace ``mindie_sim`` tree."""
    return sorted(
        path for path in (ROOT / "mindie_sim").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
        and path.suffix not in {".pyc", ".pyo"})


def _copy_source(base):
    """Copy only the package and the build declaration under ``/tmp``."""
    source = base / "source"
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")
    shutil.copytree(ROOT / "mindie_sim", source / "mindie_sim", ignore=ignore)
    shutil.copy2(ROOT / "pyproject.toml", source / "pyproject.toml")
    if (ROOT / "README.md").exists():
        shutil.copy2(ROOT / "README.md", source / "README.md")
    (source / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    return source


def _wheel_env(base):
    env = dict(os.environ)
    for name in ("MINDIE_SIM_ACCEL", "MINDIE_SIM_ENGINE_BACKEND",
                 "MINDIE_SIM_CC", "MINDIE_DATA_HOME", "XDG_CACHE_HOME",
                 "PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0",
               TMPDIR=str(base), MINDIE_CACHE_HOME=str(base / "build-cache"))
    return env


def _build_wheel(base):
    """Build the real PEP 517 universal wheel from a complete ``/tmp`` copy.

    ``setuptools.build_meta.build_wheel`` is tried first because it needs no
    build-isolation environment.  Hosts whose setuptools predates its vendored
    ``bdist_wheel`` command (this host has 65.5.0 and no ``wheel`` package)
    still get a real wheel from the isolated ``build`` frontend, which installs
    the declared ``setuptools>=64`` backend.  There is no hand-assembled
    fallback: a failed build fails the test with both captured logs.
    """
    source = _copy_source(base)
    wheel_dir = base / "dist"
    wheel_dir.mkdir()
    hook = base / "pep517_hook.py"
    hook.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "from setuptools import build_meta\n"
        "print(build_meta.build_wheel(str(Path(sys.argv[1]))))\n")
    attempts = []
    result = subprocess.run(
        [sys.executable, "-B", str(hook), str(wheel_dir)],
        cwd=source, env=_wheel_env(base), capture_output=True, text=True,
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
        [sys.executable, "-B", "-m", "build", "--wheel",
         "--outdir", str(wheel_dir), "."],
        cwd=source, env=_wheel_env(base), capture_output=True, text=True,
        timeout=600)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if result.returncode == 0 and wheels:
        return wheels[0]
    attempts.append(("python -m build --wheel", result))
    detail = "\n\n".join(
        f"[{name}] exit={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
        for name, proc in attempts)
    raise AssertionError(f"real PEP 517 wheel build failed:\n{detail}")


def _smoke_script(extract):
    return f'''
import json
import sys
from pathlib import Path
import mindie_sim
from mindie_sim.backends.ascend import profiles as ascend_profiles
from mindie_sim.backends.nvidia import profiles as nvidia_profiles
from mindie_sim.model_catalog import load_architecture_profile, load_model
from mindie_sim.prediction.engine import Engine, Resource

extract = Path(sys.argv[1]).resolve()
package = Path(mindie_sim.__file__).resolve()
assert package.is_relative_to(extract), (package, extract)
assert len(ascend_profiles()) == 15
assert len(nvidia_profiles()) == 4
assert type(load_model("qwen3-32b")).__name__ == "ModelSpec"
assert type(load_architecture_profile("longcat-flash-lite")).__name__ \\
    == "ModelArchitectureProfile"
report = {{"package": str(package)}}
for mode in ("c", "auto"):
    engine = Engine([Resource("r", 4097.0)], backend=mode, trace_limit=3)
    first = engine.submit("root0", {{"r": 1.0}})
    child = engine.submit("child", {{}}, after=(first,))
    roots = [first] + [engine.submit("root" + str(i), {{"r": 1.0}})
                       for i in range(1, 4097)]
    engine.advance(horizon=1.0)
    assert engine.now == 1.0 and engine.completed_count == 4098
    assert all(event.time == 1.0 and event.started == 0.0 for event in roots)
    assert child.started == child.time == 1.0
    assert [row["task"] for row in engine.trace] == [
        "root4095", "root4096", "child"]
    status = engine.accel_status()
    assert status["native_steps"] == 1, status
    assert status["accel_backend"] == "c_native", status
    report[mode] = {{"native_steps": status["native_steps"],
                    "completed": engine.completed_count}}
    engine._accel.close()
print(json.dumps(report))
'''


def test_r9_3_universal_wheel_ships_every_runtime_resource():
    """All workspace package files ship byte-identically and the loaders work."""
    workspace = {path.relative_to(ROOT).as_posix(): path.read_bytes()
                 for path in _package_files()}
    assert len(workspace) >= PACKAGE_FILES
    base = tempdir("nr-s3-r9-wheel-")
    try:
        wheel = _build_wheel(base)
        import mindie_sim
        ver = mindie_sim.__version__
        dist_dir = f"mindie_sim-{ver}.dist-info/"
        assert wheel.name == f"mindie_sim-{ver}-py3-none-any.whl", wheel
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
            import base64
            import csv
            import hashlib
            rec_name = dist_dir + "RECORD"
            rows = list(csv.reader(entries[rec_name].decode("utf-8").splitlines()))
            assert {r[0] for r in rows} == set(entries)
            for name, digest, size in rows:
                if name == rec_name:
                    assert digest == "" and size == ""
                    continue
                algo, expected = digest.split("=", 1)
                actual = base64.urlsafe_b64encode(
                    hashlib.new(algo, entries[name]).digest()
                ).rstrip(b"=").decode("ascii")
                assert actual == expected and int(size) == len(entries[name]), name
        package = {name: data for name, data in entries.items()
                   if name.startswith("mindie_sim/")}
        metadata = entries[dist_dir + "WHEEL"].decode()
        assert "Root-Is-Purelib: true" in metadata
        assert "Tag: py3-none-any" in metadata
        assert len(package) == len(workspace), len(package)
        dist_files = set(entries) - set(package)
        expected_dist = {dist_dir + k for k in ("METADATA", "WHEEL", "top_level.txt", "RECORD")}
        assert expected_dist <= dist_files <= (expected_dist | {dist_dir + "entry_points.txt"})
        assert set(package) == set(workspace)
        changed = [name for name in workspace
                   if package[name] != workspace[name]]
        assert changed == []
        ws_json = sum(name.endswith(".json") for name in workspace)
        assert ws_json >= 36 and sum(name.endswith(".json") for name in package) == ws_json
        for name, data in entries.items():
            assert Path(name).suffix not in PREBUILT_SUFFIXES, name
            assert not data.startswith(BINARY_MAGIC), name
        missing = {name for name in workspace if name.endswith(".json")
                   and name not in package}
        assert missing == set()
        extract = base / "installed"
        install = subprocess.run(
            [sys.executable, "-B", "-m", "pip", "install", "--no-deps",
             "--no-index", "--no-compile", "--no-cache-dir",
             "--target", str(extract), str(wheel)],
            env=_wheel_env(base), capture_output=True, text=True, timeout=240)
        assert install.returncode == 0, (install.stdout, install.stderr)
        script = base / "smoke.py"
        script.write_text(_smoke_script(extract))
        run_dir = base / "run"
        run_dir.mkdir()
        env = _wheel_env(base)
        env.update(PYTHONPATH=str(extract), MINDIE_CACHE_HOME=str(base / "cache"))
        result = subprocess.run(
            [sys.executable, "-B", str(script), str(extract)],
            cwd=run_dir, env=env, capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, (result.stdout, result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        assert report["package"].startswith(str(extract.resolve()))
        assert report["c"]["native_steps"] == 1
        assert report["auto"]["native_steps"] == 1
    finally:
        shutil.rmtree(base, ignore_errors=True)
