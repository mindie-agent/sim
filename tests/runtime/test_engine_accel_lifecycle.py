"""C11 ``_engine_accel`` build, cache, publication, drain-fault and wheel lifecycle.

[Role] Exercise the Tier-1 build/storage/process lifecycle of the engine-arena/2
  accelerator: cache-root precedence, build memo, compiler identity/digest,
  concurrent immutable publication, bounded prune and mapped-library deferral,
  failed-publication cleanup, cross-process cache lock, vanished/corrupt hit
  recovery, ``verify_wheel`` cleanup, compiled drain-allocation faults, heavy
  arena frontiers, ASan/UBSan + ``-Werror`` harness verification, and one real
  PEP 517 installed-wheel import.
[Invariants] Every test uses the session-isolated cache or a test-owned temp
  root and never writes the workspace or real user cache.  Drain faults use a
  compiled one-shot ``realloc`` wrapper around unchanged ``_engine_accel.c``
  with Python replay forbidden.
[Timeouts & Locks] Subprocesses, threads and barriers carry explicit timeouts;
  no sleeps.
[Side-Effect Policy] Writes only under ``tmp_path``, the isolated cache, or
  directories registered with ``_support.tempdirs.tempdir``.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import patch

import pytest

from _support.accel import (
    BACKENDS, BINARY_MAGIC, NATIVE, PREBUILT_SUFFIXES, ROOT,
    build_and_run_sanitizer, build_pep517_wheel, close, compile_fault_library,
    install_and_smoke_wheel, make_cache_artifact, package_files,
    verify_wheel_record,
)
from _support.tempdirs import tempdir
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.runtime import (
    accel_bridge, accel_build, accel_cache, accel_loader,
)
from mindie_sim.prediction.runtime.accel_tables import Arena, KernelFault, PROBE_GOLDEN

pytestmark = pytest.mark.tier1


def _compiler() -> str:
    compiler, reason = accel_build.discover_compiler()
    assert compiler is not None, reason or "compiler_unavailable"
    return str(Path(compiler).resolve())


@pytest.fixture(scope="module")
def fault_library():
    library, directory = compile_fault_library()
    try:
        yield library
    finally:
        shutil.rmtree(directory, ignore_errors=True)


# -- build memo, cache precedence, identity/digest & publication ------------
def test_cache_root_precedence_check_only_and_cli_failure(monkeypatch, tmp_path, capsys):
    from mindie_sim.__main__ import main

    for name in ("MINDIE_CACHE_HOME", "MINDIE_DATA_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MINDIE_CACHE_HOME", str(tmp_path / "cache-home"))
    assert accel_build.cache_root() == tmp_path / "cache-home" / "mindie" / "sim" / "engine"
    monkeypatch.delenv("MINDIE_CACHE_HOME")
    monkeypatch.setenv("MINDIE_DATA_HOME", str(tmp_path / "data"))
    assert accel_build.cache_root() == tmp_path / "data" / "cache" / "mindie" / "sim" / "engine"
    monkeypatch.delenv("MINDIE_DATA_HOME")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert accel_build.cache_root() == tmp_path / "xdg" / "mindie" / "sim" / "engine"

    monkeypatch.setenv("MINDIE_CACHE_HOME", str(tmp_path / "fresh"))
    accel_build.reset_build_memo()
    before = accel_build.build_accel(check_only=True)
    assert before["backend"] == "python_fallback" and before["library_ready"] is False
    assert accel_build.build_accel()["ok"]
    after = accel_build.build_accel(check_only=True)
    assert after["backend"] == "c_native" and after["library_ready"] is True

    monkeypatch.setenv("MINDIE_SIM_CC", "/nonexistent/cc")
    accel_build.reset_build_memo()
    missing = accel_build.build_accel(check_only=True)
    assert not missing["ok"] and not missing["library_ready"] and not missing["toolchain_ready"]
    code = main(["build-accel"])
    assert code != 0 and json.loads(capsys.readouterr().out)["ok"] is False
    accel_build.reset_build_memo()


def test_compiler_identity_and_digest_ignore_path_and_track_flags(monkeypatch):
    compiler = _compiler()
    monkeypatch.setenv("MINDIE_SIM_CC", compiler)
    monkeypatch.setenv("SDKROOT", "")
    source, flags = accel_build.source_bytes(), accel_build.compiler_flags()
    orig_path, cdir = os.environ.get("PATH", ""), str(Path(compiler).parent)
    p1, p2 = os.pathsep.join((cdir, orig_path)), os.pathsep.join((orig_path, cdir))
    assert p1 != p2

    monkeypatch.setenv("PATH", p1)
    accel_build.reset_build_memo()
    id1 = accel_build._compiler_identity(compiler)
    assert id1.split("|", 1)[0] == compiler
    d1 = accel_build.cache_digest(source, compiler, flags, id1)
    built = accel_build.build_accel()
    assert built["ok"], built.get("detail") or built.get("reason")

    monkeypatch.setenv("PATH", p2)
    accel_build.reset_build_memo()
    id2 = accel_build._compiler_identity(compiler)
    assert id1 == id2 and d1 == accel_build.cache_digest(source, compiler, flags, id2)
    hit = accel_build.build_accel()
    assert hit["ok"] and hit["cache_hit"] is True and hit["cache_path"] == built["cache_path"]
    assert d1 != accel_build.cache_digest(source, compiler, flags + ["-DMSIM_X"], id1)


def test_build_memo_force_rebuild_and_golden_probe_validation(monkeypatch):
    accel_build.reset_build_memo()
    first = accel_build.build_accel()
    second = accel_build.build_accel(force=True)
    assert first["ok"] and second["ok"] and first["cache_path"] != second["cache_path"]
    assert Path(first["cache_path"]).is_file() and Path(second["cache_path"]).is_file()

    monkeypatch.setattr(accel_build, "_compiler_identity",
                        lambda _c, *, persist=False: "replaced-toolchain")
    third = accel_build.build_accel()
    assert third["ok"] and third["cache_path"] != second["cache_path"]
    Path(third["cache_path"]).unlink()
    fourth = accel_build.build_accel()
    assert fourth["ok"] and Path(fourth["cache_path"]).is_file() and fourth["cache_hit"] is False

    class _Probe:
        returncode, stderr = 0, ""

        def __init__(self, cells):
            self.stdout = json.dumps({"v": 2, "rc": 0, "cells": cells})

    monkeypatch.setattr(accel_build.subprocess, "run", lambda *_a, **_kw: _Probe([0] * 16))
    assert accel_build._validate(Path("ignored")) == (False, "abi_probe_mismatch")
    monkeypatch.setattr(accel_build.subprocess, "run",
                        lambda *_a, **_kw: _Probe([int(v) for v in PROBE_GOLDEN]))
    assert accel_build._validate(Path("ignored")) == (True, "")


def test_concurrent_publication_produces_distinct_immutable_inodes(monkeypatch):
    original, barrier, results = accel_build._validate, threading.Barrier(2), {}

    def synchronized(path):
        result = original(path)
        barrier.wait(timeout=120)
        return result

    monkeypatch.setattr(accel_build, "_validate", synchronized)
    accel_build.reset_build_memo()
    threads = [threading.Thread(
        target=lambda i=idx: results.__setitem__(i, accel_build.build_accel(force=True)))
        for idx in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)
    assert len(results) == 2 and all(s["ok"] for s in results.values())
    paths = [Path(results[i]["cache_path"]) for i in range(2)]
    assert paths[0] != paths[1] and paths[0].stat().st_ino != paths[1].stat().st_ino
    assert all(p.is_file() and p.stat().st_size > 0 for p in paths)


# -- prune, lock, manifest, failed publication & hit recovery ---------------
def test_prune_bounds_artifacts_manifests_markers_and_stale_temporaries(monkeypatch, tmp_path):
    cache, ext = tmp_path / "engine", accel_build.library_extension()
    cache.mkdir()
    artifacts = [make_cache_artifact(cache, i, ext, 1000 - i) for i in range(30)]
    newest = artifacts[-1][0]
    for i in range(20):
        marker = cache / f"compiler-{i:016x}.id"
        marker.write_text("id")
        os.utime(marker, (1000 - i, 1000 - i))
    removed = accel_build._prune_engine_cache(cache, keep_key=newest, max_artifacts=16)
    survivors = accel_build._artifact_paths(cache)
    assert removed and len(survivors) <= 16 and newest.is_file() and not artifacts[0][0].exists()
    assert len(list(cache.glob("compiler-*.id"))) == 16
    assert all(Path(f"{lib}.json").is_file() for lib in survivors)

    # Empty-of-artifacts cache still sweeps stale src/build/trash/tmp while sparing fresh entries.
    sweep, old = tmp_path / "sweep", time.time() - 7200
    sweep.mkdir()
    stale_dir, fresh_dir = sweep / "src-1-2-3", sweep / "src-fresh"
    stale_dir.mkdir()
    fresh_dir.mkdir()
    (stale_dir / "x.c").write_text("/* stale */")
    os.utime(stale_dir, (old, old))
    stale_files = [sweep / "id-stale.tmp", sweep / f"build-1-2-3{ext}",
                   sweep / ".trash-1-old.so.tmp"]
    fresh_files = [sweep / "m-fresh.tmp", sweep / f"build-fresh{ext}",
                   sweep / ".trash-2-fresh.so.tmp"]
    for p in stale_files:
        p.write_bytes(b"stale")
        os.utime(p, (old, old))
    for p in fresh_files:
        p.write_bytes(b"fresh")
    accel_build._prune_engine_cache(sweep)
    assert not stale_dir.exists() and all(not p.exists() for p in stale_files)
    assert fresh_dir.is_dir() and all(p.is_file() for p in fresh_files)

    # Numeric manifest generation sorting (9 < 10) + malformed non-object removal.
    monkeypatch.setenv("MINDIE_CACHE_HOME", str(tmp_path / "gen"))
    root, digest = accel_build.cache_root(), "a" * 64
    root.mkdir(parents=True)
    older, newer = root / f"{digest}-9{ext}", root / f"{digest}-10{ext}"
    for lib in (older, newer):
        lib.write_bytes(b"native")
        (root / f"{lib.name}.json").write_text(json.dumps({"digest": digest, "library": str(lib)}))
    malformed = root / f"{digest}-11{ext}.json"
    malformed.write_text("[1]")
    assert accel_build._read_manifest(digest) == newer
    accel_build._prune_engine_cache(root)
    assert not malformed.exists() and older.is_file() and newer.is_file()


def test_prune_defers_mapped_library_and_cache_hit_reclaims_duplicates(monkeypatch, tmp_path):
    cache, ext = tmp_path / "defer", accel_build.library_extension()
    cache.mkdir()
    artifacts = [make_cache_artifact(cache, i, ext, 1000 - i) for i in range(20)]
    blocked, orig = artifacts[0][0], accel_build.os.replace
    with monkeypatch.context() as mp:
        mp.setattr(accel_build.os, "replace", lambda s, d: (
            (_ for _ in ()).throw(PermissionError("mapped")) if Path(s) == blocked else orig(s, d)))
        accel_build._prune_engine_cache(cache, keep_key=artifacts[-1][0], max_artifacts=16)
    assert blocked.is_file() and Path(f"{blocked}.json").is_file()
    assert len(accel_build._artifact_paths(cache)) <= 16

    monkeypatch.setenv("MINDIE_SIM_CC", _compiler())
    monkeypatch.setenv("MINDIE_CACHE_HOME", str(tmp_path / "reclaim"))
    accel_build.reset_build_memo()
    built = accel_build.build_accel()
    root = accel_build.cache_root()
    for i in range(20):
        make_cache_artifact(root, 0xA000 + i, ext, 1000 - i)
    assert built["ok"] and len(accel_build._artifact_paths(root)) == 21
    accel_build.reset_build_memo()
    hit = accel_build.build_accel()
    assert hit["cache_hit"] is True and hit["cache_path"] == built["cache_path"]
    assert len(accel_build._artifact_paths(root)) <= 16


@pytest.mark.parametrize("mode", ("compile_fail", "compile_timeout", "manifest_fail"))
def test_failed_publication_leaves_no_partial_artifacts(tmp_path, monkeypatch, mode):
    cache = tmp_path / "engine"
    files = {name: b"/* test */\n" for name in accel_build.SOURCE_NAMES}

    class _Done:
        stdout, stderr, returncode = "", "synthetic failure", 1 if mode == "compile_fail" else 0

    if mode == "compile_timeout":
        monkeypatch.setattr(accel_build.subprocess, "run", lambda *_a, **_kw: (
            _ for _ in ()).throw(subprocess.TimeoutExpired(cmd="cc", timeout=20)))
    else:
        monkeypatch.setattr(accel_build.subprocess, "run", lambda *_a, **_kw: _Done())
    if mode == "manifest_fail":
        orig = accel_build.os.replace
        monkeypatch.setattr(accel_build, "_validate", lambda _p: (True, ""))
        monkeypatch.setattr(accel_build.os, "replace", lambda s, d: (
            (_ for _ in ()).throw(OSError("denied")) if str(d).endswith(".json") else orig(s, d)))
    ok, detail, candidate = accel_build._publish("cc", ["-shared"], cache, files, "a" * 64)
    assert not ok and candidate is None
    if mode == "compile_timeout":
        assert detail == "accel_compile_timeout"
    elif mode == "manifest_fail":
        assert "publication_failed" in detail
    assert [p.name for p in cache.iterdir() if p.name != accel_cache.LOCK_NAME] == []


def test_locked_loader_survives_seventeenth_artifact_and_excludes_other_process(
        monkeypatch, tmp_path):
    monkeypatch.setenv("MINDIE_SIM_CC", _compiler())
    monkeypatch.setenv("MINDIE_CACHE_HOME", str(tmp_path / "cache"))
    accel_build.reset_build_memo()
    built = accel_build.build_accel()
    cache, ext, hit = accel_build.cache_root(), accel_build.library_extension(), Path(built["cache_path"])
    old = time.time() - 100
    os.utime(hit, (old, old))
    for i in range(15):
        make_cache_artifact(cache, 0xB000 + i, ext, 0)
    assert built["ok"] and min(accel_build._artifact_paths(cache), key=accel_cache._mtime_ns) == hit

    at_map, allow_map, attempted, acquired = (threading.Event() for _ in range(4))
    real_cdll, loaded, evicted, failures = accel_bridge.ctypes.CDLL, {}, {}, []

    def delayed_cdll(path):
        at_map.set()
        assert allow_map.wait(10.0)
        return real_cdll(path)

    monkeypatch.setattr(accel_bridge.ctypes, "CDLL", delayed_cdll)

    def load_hit():
        try:
            loaded["value"] = accel_loader.load_accel_cdll()
        except BaseException as err:
            failures.append(err)

    def publish_17th():
        attempted.set()
        with accel_cache.cache_lock(cache):
            acquired.set()
            newest, _ = make_cache_artifact(cache, 0xC000, ext, 0)
            evicted["removed"] = accel_build._prune_engine_cache(cache, keep_key=newest, max_artifacts=16)

    t1, t2 = threading.Thread(target=load_hit), threading.Thread(target=publish_17th)
    t1.start()
    assert at_map.wait(10.0)
    t2.start()
    try:
        assert attempted.wait(10.0) and not acquired.wait(0.2)
        allow_map.set()
        t1.join(30.0)
        t2.join(30.0)
    finally:
        allow_map.set()
    assert not failures and not t1.is_alive() and not t2.is_alive()
    status, library = loaded["value"]
    assert status["cache_path"] == str(hit) and int(library.msim_abi_version()) == 2
    assert hit.is_file() and hit.stat().st_mtime_ns > int(old * 1e9)
    assert not any(p.name == hit.name for p in evicted.get("removed", []))

    lock_dir = tmp_path / "xproc"
    lock_dir.mkdir()
    script = ("import sys; from pathlib import Path; from mindie_sim.prediction.runtime import accel_cache\n"
              "try:\n  with accel_cache.cache_lock(Path(sys.argv[1]), timeout=0.2): print('acquired')\n"
              "except TimeoutError: print('timeout')\n")
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    with accel_cache.cache_lock(lock_dir):
        r1 = subprocess.run([sys.executable, "-c", script, str(lock_dir)],
                            capture_output=True, text=True, timeout=60, env=env)
        assert r1.returncode == 0 and r1.stdout.strip() == "timeout", r1.stderr
    r2 = subprocess.run([sys.executable, "-c", script, str(lock_dir)],
                        capture_output=True, text=True, timeout=60, env=env)
    assert r2.returncode == 0 and r2.stdout.strip() == "acquired", r2.stderr


def test_load_library_recovers_vanished_and_corrupt_hits_and_verify_wheel_cleans(
        monkeypatch, tmp_path):
    monkeypatch.setenv("MINDIE_SIM_CC", _compiler())
    monkeypatch.setenv("MINDIE_CACHE_HOME", str(tmp_path / "rec"))
    accel_build.reset_build_memo()
    first = accel_build.build_accel(force=True)
    assert first["ok"]
    Path(first["cache_path"]).unlink()
    assert int(accel_bridge.load_library(first).msim_abi_version()) == 2
    assert Path(first["cache_path"]).is_file()

    second = accel_build.build_accel(force=True)
    assert second["ok"]
    corrupt = Path(second["cache_path"])
    corrupt.write_bytes(b"not a shared library")
    assert int(accel_bridge.load_library(second).msim_abi_version()) == 2
    assert corrupt.read_bytes() == b"not a shared library"
    assert len(accel_build._artifact_paths(accel_build.cache_root())) >= 2

    spec = importlib.util.spec_from_file_location("_verify_wheel", ROOT / "tools" / "verify_wheel.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scratch = tmp_path / "vw-tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    monkeypatch.setattr(mod.subprocess, "run", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError):
        mod.verify(str(tmp_path / "w.whl"), str(ROOT), str(tmp_path / "r.json"))
    assert list(scratch.iterdir()) == []


# -- compiled drain-allocation faults, heavy frontiers, sanitizer & wheel ---
@pytest.mark.parametrize("scenario", ("admission", "completion", "no-prefix"))
@pytest.mark.parametrize("backend", NATIVE)
def test_drain_allocation_failure_publishes_accepted_prefix(fault_library, scenario, backend):
    orig, armed = Arena.run, [False]

    def run(self, now, horizon, next_release, budget):
        if budget == 0 and not armed[0] and (scenario == "admission" or now == 1.0):
            fault_library.drain_fault_arm(self.ctx, {"admission": 1, "completion": 2, "no-prefix": 3}[scenario])
            armed[0] = True
        return orig(self, now, horizon, next_release, budget)

    with patch.object(accel_bridge, "load_library", lambda _s: fault_library), \
            patch.object(accel_build, "build_accel", lambda **_kw: {"ok": True, "cache_path": "fault"}), \
            patch.object(Arena, "run", run):
        eng = Engine([Resource("r", 1.0 if scenario == "no-prefix" else 4097.0)], backend=backend)
        before = fault_library.drain_fault_count()
        if scenario == "admission":
            evs, act = [eng.submit(f"z{i}", {}) for i in range(4097)], lambda: eng.advance(horizon=0.0)
        elif scenario == "completion":
            evs = [eng.submit(f"root{i}", {"r": 1.0}) for i in range(4097)]
            for i in range(8193):
                eng.submit(f"c{i}", {"r": 1.0}, after=(evs[-1],))
            eng.advance(horizon=0.0)
            act = lambda: eng.advance(horizon=1.0)
        else:
            good = eng.submit("good", {"r": 0.5})
            evs = [eng.submit(f"late{i}", {"r": 1.0}, release=1.0) for i in range(4097)]
            eng.advance(horizon=0.0)
            act = lambda: eng.advance(horizon=1.0)
        with patch.object(Engine, "_rates", side_effect=AssertionError("Python replay forbidden")):
            with pytest.raises(KernelFault) as info:
                act()
        assert fault_library.drain_fault_count() == before + 1 and info.value.code == 1
        if scenario == "admission":
            assert info.value.has_prefix and info.value.prefix == (0.0, 0, 0, 0, 1) and all(e.started == 0.0 for e in evs)
            assert (eng.now, eng.advance_steps, eng.completed_count) == (0.0, 0, 0)
            assert eng.accel_status()["native_calls"] == 2
        elif scenario == "completion":
            assert info.value.has_prefix and info.value.prefix == (1.0, 0, 0, 1, 0)
            assert (eng.now, eng.advance_steps, eng.completed_count) == (1.0, 1, 4097)
            assert evs[-1].time == 1.0 and eng.busy_ms["r"] == 1.0 and eng.serviced["r"] == 4097.0
        else:
            assert not info.value.has_prefix and info.value.prefix is None and good.time == 0.5
            assert (eng.now, eng.advance_steps) == (1.0, 2) and sum(e.started is not None for e in evs) == 4096
            assert eng.busy_ms["r"] == 0.5 and eng.serviced["r"] == 0.5
        close(eng)


def test_static_task_link_solver_and_implicit_unit_frontiers():
    eng_t_max = Engine([Resource("r", 65536.0)], backend="c")
    [eng_t_max.submit(f"t{i}", {"r": 1.0}) for i in range(65536)]
    eng_t_max.advance()
    assert eng_t_max.accel_status()["fallback_reason"] is None and eng_t_max.accel_status()["native_steps"] > 0
    close(eng_t_max)

    ns32 = [f"r{i}" for i in range(32)]
    eng_l_max = Engine([Resource(n, 8192.0) for n in ns32], backend="c")
    [eng_l_max.submit(f"t{i}", dict.fromkeys(ns32, 1.0)) for i in range(8192)]
    eng_l_max.advance()
    assert eng_l_max.accel_status()["fallback_reason"] is None and eng_l_max.accel_status()["native_steps"] > 0
    close(eng_l_max)

    for make in (
        lambda: (e := Engine([Resource("r", 1e6)], backend="c"), [e.submit(f"t{i}", {}) for i in range(65537)])[0],
        lambda: (e := Engine([Resource(n, 1000.0) for n in ns32], backend="c"),
                 [e.submit(f"t{i}", dict.fromkeys(ns32, 1.0)) for i in range(8192)],
                 e.submit("extra", {"r0": 1.0}))[0],
        lambda: (e := Engine([Resource(n, 100.0) for n in ns32], backend="c"),
                 [e.submit(f"c{i}", dict.fromkeys(ns32, 1.0), coupled=True) for i in range(4096)])[0],
        lambda: (e := Engine([Resource("r", 64.0, units=64)], backend="c"),
                 [e.submit(f"p{i}", {"r": 1.0}, placement={"r": tuple(range(64))}) for i in range(32769)])[0],
    ):
        eng = make()
        assert eng.accel_status()["fallback_reason"] == "capacity_exceeded"
        assert eng.accel_status()["native_steps"] == 0
        close(eng)

    units, out = tuple(range(64)), {}
    for b in BACKENDS:
        eng = Engine([Resource("r", 64.0, units=64)], backend=b)
        evs = [eng.submit(str(i), {"r": 1.0}, placement={"r": units}, coupled=True) for i in range(8193)]
        eng.advance()
        out[b] = (eng.now, eng.completed_count, evs[-1].time, eng.accel_status()["native_steps"])
        close(eng)
    for b in NATIVE:
        assert out[b][:3] == out["python"][:3] and out[b][3] > 0

    drained = {}
    for b in BACKENDS:
        eng = Engine([Resource("r", 64.0, units=64)], backend=b)
        warm = [eng.submit(f"w{i}", {"r": 1.0}) for i in range(64)]
        eng.advance(horizon=0.5)
        midpoint = (eng.now, eng.advance_steps, eng.accel_status()["native_steps"])
        batch = [eng.submit(f"d{i}", {"r": 1.0}) for i in range(32704)]
        pinned = eng.submit("pinned", {"r": 1.0}, placement={"r": tuple(range(63))})
        eng.advance(horizon=1.0)
        mid1 = (eng.now, eng.completed_count, eng.serviced["r"])
        eng.advance()
        status = eng.accel_status()
        drained[b] = (
            midpoint, mid1, eng.now, eng.completed_count, dict(eng.serviced), dict(eng.busy_ms),
            eng.advance_steps, warm[0].time, pinned.time, batch[0].time, batch[-1].time,
            status["fallback_reason"],
        )
        close(eng)
    assert drained["python"][0] == (0.5, 1, 0)
    assert (drained["python"][2], drained["python"][3], drained["python"][5]) == (
        512.015873015873, 32769, {"r": 512.015873015873})
    assert drained["python"][4]["r"] == pytest.approx(32769.0, abs=1e-7)
    for b in NATIVE:
        assert drained[b][0][:2] == drained["python"][0][:2] and drained[b][0][2] > 0, b
        assert drained[b][1:11] == drained["python"][1:11], b
        assert drained[b][2] == 512.015873015873 and drained[b][11] == "capacity_exceeded", b


def test_native_source_and_sanitizer_harness_pass_asan_ubsan_and_werror(tmp_path):
    ran, strict = build_and_run_sanitizer(tmp_path)
    combined = ran.stdout + ran.stderr
    assert ran.returncode == 0 and "sanitizer_harness: ok" in ran.stdout, combined[-3000:]
    assert "AddressSanitizer" not in combined and "runtime error" not in combined
    for done in strict:
        assert done.returncode == 0, done.stderr[-2000:]


def test_installed_pep517_wheel_ships_runtime_resources_and_runs_native():
    import mindie_sim

    workspace = {p.relative_to(ROOT).as_posix(): p.read_bytes() for p in package_files()}
    assert len(workspace) >= 490
    base = tempdir("accel-wheel-")
    try:
        wheel = build_pep517_wheel(base)
        ver = mindie_sim.__version__
        dist_dir = f"mindie_sim-{ver}.dist-info/"
        assert wheel.name == f"mindie_sim-{ver}-py3-none-any.whl"
        entries = verify_wheel_record(wheel)
        package = {k: v for k, v in entries.items() if k.startswith("mindie_sim/")}
        data_prefix = f"mindie_sim-{ver}.data/data/data/primitives/"
        data_files = {k for k in entries if k.startswith(data_prefix)}
        assert data_prefix + "manifest.json" in data_files
        assert any(k.endswith(".capture.json") for k in data_files)
        dist_files = set(entries) - set(package) - data_files
        expected_dist = {dist_dir + k for k in ("METADATA", "WHEEL", "top_level.txt", "RECORD")}
        assert expected_dist <= dist_files <= (expected_dist | {dist_dir + "entry_points.txt"})
        meta = entries[dist_dir + "WHEEL"].decode()
        assert "Root-Is-Purelib: true" in meta and "Tag: py3-none-any" in meta
        assert set(package) == set(workspace) and all(package[k] == workspace[k] for k in workspace)
        for name, data in entries.items():
            assert Path(name).suffix not in PREBUILT_SUFFIXES and not data.startswith(BINARY_MAGIC), name
        extract, report = install_and_smoke_wheel(base, wheel)
        assert report["package"].startswith(str(extract.resolve()))
        assert report["primitives_dir"].startswith(str(extract.resolve()))
        assert report["primitive_tables"] >= 1
        assert report["c"] == 1 and report["auto"] == 1 and report["stable"] is False
        for mode in NATIVE:
            assert report[f"desc_{mode}"] == report["desc_python"], mode
        userbase = base / "userbase"
        user_env = {**os.environ, "PYTHONUSERBASE": str(userbase), "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
        inst_u = subprocess.run(
            [sys.executable, "-B", "-m", "pip", "install", "--user", "--no-deps", "--no-index",
             "--no-compile", "--no-cache-dir", str(wheel)],
            env=user_env, capture_output=True, text=True, timeout=240)
        assert inst_u.returncode == 0, (inst_u.stdout, inst_u.stderr)
        probe_u = subprocess.run(
            [sys.executable, "-B", "-c",
             "from pathlib import Path; from mindie_sim.primitives.loader import default_primitives_dir, load_manifest;"
             " d = default_primitives_dir().resolve(); m = load_manifest();"
             " print(d); assert m.get('tables')"],
            cwd=base / "run", env={k: v for k, v in user_env.items() if k != "PYTHONPATH"},
            capture_output=True, text=True, timeout=60)
        assert probe_u.returncode == 0, (probe_u.stdout, probe_u.stderr)
        assert Path(probe_u.stdout.strip()).is_relative_to(userbase.resolve())
    finally:
        shutil.rmtree(base, ignore_errors=True)
