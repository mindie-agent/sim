"""Storage/temporary-directory lifecycle regressions for v0.9.15 (S-1..S-4).

[Role] Prove that compiler cache identities ignore the caller's raw PATH, that
  engine retention is bounded and defers mapped-library eviction, that failed
  publications leave no partial artifacts, and that wheel/probe/budget helpers
  remove their temporary workspaces.
[Invariants] Cache assertions use isolated cache roots; PATH changes resolve the
  same real compiler; the prune test uses real files/manifests and a real
  ``PermissionError`` path; no test writes the user's HOME or real cache.
[Timeouts & Locks] At most one real C compile and one bounded two-thread probe;
  every subprocess carries a timeout and temporary cleanup is in ``finally``.
[Side-Effect Policy] Writes only pytest-owned or isolated session cache paths;
  removes every workspace it creates and registers external temp roots.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import tomllib

import pytest

import mindie_sim
from mindie_sim.prediction.runtime import (
    accel_bridge, accel_build, accel_cache, accel_loader,
)

ROOT = Path(__file__).resolve().parents[1]


def _tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_storage_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_storage_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compiler_or_skip() -> str:
    compiler, reason = accel_build.discover_compiler()
    if compiler is None:
        pytest.skip(reason or "compiler_unavailable")
    return str(Path(compiler).resolve())


def _artifact(cache: Path, index: int, extension: str, age: float):
    digest = f"{index:064x}"
    library = cache / f"{digest}-0{extension}"
    library.write_bytes(b"native")
    manifest = cache / (library.name + ".json")
    manifest.write_text(json.dumps({"digest": digest,
                                    "library": str(library)}))
    stamp = time.time() - age
    os.utime(library, (stamp, stamp))
    os.utime(manifest, (stamp, stamp))
    return library, manifest


def test_compiler_identity_and_digest_ignore_virtualenv_path(monkeypatch):
    compiler = _compiler_or_skip()
    monkeypatch.setenv("MINDIE_SIM_CC", compiler)
    monkeypatch.setenv("SDKROOT", "")
    source = accel_build.source_bytes()
    flags = accel_build.compiler_flags()
    original_path = os.environ.get("PATH", "")
    compiler_dir = str(Path(compiler).parent)
    first_path = os.pathsep.join((compiler_dir, original_path))
    second_path = os.pathsep.join((original_path, compiler_dir))
    assert first_path != second_path

    monkeypatch.setenv("PATH", first_path)
    accel_build.reset_build_memo()
    first_identity = accel_build._compiler_identity(compiler)
    first_digest = accel_build.cache_digest(source, compiler, flags,
                                            first_identity)

    monkeypatch.setenv("PATH", second_path)
    accel_build.reset_build_memo()
    second_identity = accel_build._compiler_identity(compiler)
    second_digest = accel_build.cache_digest(source, compiler, flags,
                                             second_identity)

    assert first_identity == second_identity
    assert first_digest == second_digest

    monkeypatch.setenv("PATH", first_path)
    accel_build.reset_build_memo()
    built = accel_build.build_accel(force=True)
    assert built["ok"], built.get("detail") or built.get("reason")
    monkeypatch.setenv("PATH", second_path)
    accel_build.reset_build_memo()
    hit = accel_build.build_accel()
    assert hit["ok"] and hit["cache_hit"] is True
    assert hit["cache_path"] == built["cache_path"]


def test_cache_hit_reclaims_historical_duplicates(monkeypatch, tmp_path):
    compiler = _compiler_or_skip()
    monkeypatch.setenv("MINDIE_SIM_CC", compiler)
    monkeypatch.setenv("MINDIE_CACHE_HOME", str(tmp_path / "cache"))
    accel_build.reset_build_memo()
    built = accel_build.build_accel()
    assert built["ok"], built.get("detail") or built.get("reason")
    cache = accel_build.cache_root()
    extension = accel_build.library_extension()
    for index in range(20):
        _artifact(cache, 0xA000 + index, extension, 1000 - index)
    assert len(accel_build._artifact_paths(cache)) == 21

    accel_build.reset_build_memo()
    hit = accel_build.build_accel()
    assert hit["cache_hit"] is True
    assert hit["cache_path"] == built["cache_path"]
    assert Path(built["cache_path"]).is_file()
    assert len(accel_build._artifact_paths(cache)) <= 16


def test_cache_digest_tracks_semantic_flags():
    compiler = _compiler_or_skip()
    source = accel_build.source_bytes()
    flags = accel_build.compiler_flags()
    base = accel_build.cache_digest(source, compiler, flags, "toolchain")
    changed = accel_build.cache_digest(source, compiler, flags + ["-DMSIM_X"],
                                       "toolchain")
    assert base != changed


def test_prune_engine_cache_bounds_artifacts_manifests_and_markers(tmp_path):
    cache = tmp_path / "engine"
    cache.mkdir()
    extension = accel_build.library_extension()
    artifacts = [_artifact(cache, index, extension, 1000 - index)
                 for index in range(30)]
    newest = artifacts[-1][0]
    markers = []
    for index in range(20):
        marker = cache / f"compiler-{index:016x}.id"
        marker.write_text("identity")
        stamp = 1000 - index
        os.utime(marker, (stamp, stamp))
        markers.append(marker)
    stale = cache / "identity-stale.tmp"
    stale.write_text("stale")
    os.utime(stale, (time.time() - 7200, time.time() - 7200))
    fresh = cache / "manifest-fresh.tmp"
    fresh.write_text("fresh")

    removed = accel_build._prune_engine_cache(
        cache, keep_key=newest, max_artifacts=16)

    survivors = accel_build._artifact_paths(cache)
    assert len(survivors) <= 16
    assert newest.is_file()
    assert removed
    assert len(list(cache.glob("compiler-*.id"))) == 16
    assert not stale.exists()
    assert fresh.exists()
    for library in survivors:
        assert Path(f"{library}.json").is_file()
    assert not artifacts[0][0].exists()


def test_prune_engine_cache_defers_a_mapped_library(tmp_path, monkeypatch):
    cache = tmp_path / "engine"
    cache.mkdir()
    extension = accel_build.library_extension()
    artifacts = [_artifact(cache, index, extension, 1000 - index)
                 for index in range(20)]
    blocked = artifacts[0][0]
    original_replace = accel_build.os.replace

    def guarded_replace(source, destination):
        if Path(source) == blocked:
            raise PermissionError("library is mapped by a running process")
        return original_replace(source, destination)

    monkeypatch.setattr(accel_build.os, "replace", guarded_replace)
    accel_build._prune_engine_cache(
        cache, keep_key=artifacts[-1][0], max_artifacts=16)

    assert blocked.is_file()
    assert Path(f"{blocked}.json").is_file()
    assert len(accel_build._artifact_paths(cache)) <= 16


class _Completed:
    stdout = ""
    stderr = ""
    returncode = 0


def test_failed_compile_removes_snapshot_and_build_temporary(tmp_path,
                                                             monkeypatch):
    cache = tmp_path / "engine"
    files = {name: b"/* test */\n" for name in accel_build.SOURCE_NAMES}
    failed = _Completed()
    failed.returncode = 1
    failed.stderr = "synthetic compiler failure"
    monkeypatch.setattr(accel_build.subprocess, "run",
                        lambda *_args, **_kwargs: failed)
    ok, _detail, candidate = accel_build._publish(
        "cc", ["-shared"], cache, files, "a" * 64)
    assert not ok and candidate is None
    assert list(cache.iterdir()) == []


def test_compile_timeout_removes_snapshot_and_build_temporary(tmp_path,
                                                              monkeypatch):
    cache = tmp_path / "engine"
    files = {name: b"/* test */\n" for name in accel_build.SOURCE_NAMES}

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="cc", timeout=20)

    monkeypatch.setattr(accel_build.subprocess, "run", timeout)
    ok, detail, candidate = accel_build._publish(
        "cc", ["-shared"], cache, files, "c" * 64)
    assert not ok and detail == "accel_compile_timeout" and candidate is None
    assert list(cache.iterdir()) == []


def test_failed_manifest_publication_removes_candidate_and_temporaries(
        tmp_path, monkeypatch):
    cache = tmp_path / "engine"
    files = {name: b"/* test */\n" for name in accel_build.SOURCE_NAMES}
    monkeypatch.setattr(accel_build.subprocess, "run",
                        lambda *_args, **_kwargs: _Completed())
    monkeypatch.setattr(accel_build, "_validate", lambda _path: (True, ""))
    original_replace = accel_build.os.replace

    def fail_manifest(source, destination):
        if str(destination).endswith(".json"):
            raise OSError("manifest publication denied")
        return original_replace(source, destination)

    monkeypatch.setattr(accel_build.os, "replace", fail_manifest)
    ok, detail, candidate = accel_build._publish(
        "cc", ["-shared"], cache, files, "b" * 64)
    assert not ok and candidate is None
    assert "publication_failed" in detail
    assert [path.name for path in cache.iterdir()
            if path.name != accel_cache.LOCK_NAME] == []


def test_verify_wheel_removes_its_workspace_on_failure(tmp_path, monkeypatch):
    module = _tool("verify_wheel")
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic venv failure")

    monkeypatch.setattr(module.subprocess, "run", fail)
    with pytest.raises(RuntimeError):
        module.verify(str(tmp_path / "wheel.whl"), str(ROOT),
                      str(tmp_path / "report.json"))
    assert list(scratch.iterdir()) == []


def test_round4_probe_removes_its_workspace(tmp_path, monkeypatch):
    pytest.importorskip("mindie_profiler")
    from mindie_profiler.fabric_probe import build_fabric_probe_plan
    from nr_s6_round4_probe import run_ranks

    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    plan = build_fabric_probe_plan(
        world_size=2, ranks_per_node=2, sizes=[4096], ops=["all_reduce"],
        iterations=1, warmup=0, duration_s=60.0,
        topology={"tiers": [{"tier": "intra_node_hccs", "ranks": [0, 1],
                             "protocol": "HCCS"}]})
    world, results = run_ranks(plan)
    assert world.errors == {}
    assert set(results) == {0, 1}
    assert list(scratch.iterdir()) == []



def test_cache_hit_refresh_and_locked_loader_survive_seventeenth_artifact(
        monkeypatch, tmp_path):
    """A hit becomes newest and stays mapped while a 17th artifact is pruned."""
    compiler = _compiler_or_skip()
    monkeypatch.setenv("MINDIE_SIM_CC", compiler)
    monkeypatch.setenv("MINDIE_CACHE_HOME", str(tmp_path / "cache"))
    accel_build.reset_build_memo()
    built = accel_build.build_accel(force=True)
    assert built["ok"], built.get("detail") or built.get("reason")
    cache = accel_build.cache_root()
    extension = accel_build.library_extension()
    hit = Path(built["cache_path"])
    old = time.time() - 100
    os.utime(hit, (old, old))
    for index in range(15):
        _artifact(cache, 0xB000 + index, extension, 0)
    assert min(accel_build._artifact_paths(cache),
               key=accel_cache._mtime_ns) == hit

    at_map = threading.Event()
    allow_map = threading.Event()
    real_cdll = accel_bridge.ctypes.CDLL

    def delayed_cdll(path):
        at_map.set()
        assert allow_map.wait(10.0)
        return real_cdll(path)

    monkeypatch.setattr(accel_bridge.ctypes, "CDLL", delayed_cdll)
    loaded = {}
    failures = []

    def load_hit():
        try:
            loaded["value"] = accel_loader.load_accel_cdll()
        except BaseException as error:  # pragma: no cover - failure detail
            failures.append(error)

    thread = threading.Thread(target=load_hit, name="locked-loader")
    thread.start()
    assert at_map.wait(10.0)
    attempted = threading.Event()
    acquired = threading.Event()
    evicted = {}

    def publish_seventeenth():
        attempted.set()
        with accel_cache.cache_lock(cache):
            acquired.set()
            newest, _manifest = _artifact(cache, 0xC000, extension, 0)
            evicted["removed"] = accel_build._prune_engine_cache(
                cache, keep_key=newest, max_artifacts=16)

    publisher = threading.Thread(target=publish_seventeenth,
                                 name="publisher")
    publisher.start()
    try:
        assert attempted.wait(10.0)
        # The loader holds the lock across CDLL, so the publisher cannot prune
        # the hit while it is being mapped.
        assert not acquired.wait(0.2)
        allow_map.set()
        thread.join(30.0)
        publisher.join(30.0)
    finally:
        allow_map.set()

    assert not failures, failures
    assert not thread.is_alive() and not publisher.is_alive()
    status, library = loaded["value"]
    assert status["cache_path"] == str(hit)
    assert int(library.msim_abi_version()) == 2
    assert hit.is_file()
    assert hit.stat().st_mtime_ns > int(old * 1e9)
    assert not any(path.name == hit.name
                   for path in evicted.get("removed", []))


def test_cache_lock_excludes_another_process(tmp_path):
    """A held lock is visible cross-process, not only between threads."""
    cache = tmp_path / "engine"
    cache.mkdir()
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from mindie_sim.prediction.runtime import accel_cache\n"
        "try:\n"
        "    with accel_cache.cache_lock(Path(sys.argv[1]), timeout=0.2):\n"
        "        print('acquired')\n"
        "except TimeoutError:\n"
        "    print('timeout')\n"
    )

    def probe() -> str:
        completed = subprocess.run(
            [sys.executable, "-c", script, str(cache)],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PYTHONPATH": str(ROOT)})
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip()

    with accel_cache.cache_lock(cache):
        assert probe() == "timeout"
    assert probe() == "acquired"


def test_load_library_recovers_vanished_and_corrupt_hits(monkeypatch, tmp_path):
    """A stale standalone status rebuilds under the loader's cache lock."""
    compiler = _compiler_or_skip()
    monkeypatch.setenv("MINDIE_SIM_CC", compiler)
    monkeypatch.setenv("MINDIE_CACHE_HOME", str(tmp_path / "cache"))
    accel_build.reset_build_memo()
    first = accel_build.build_accel(force=True)
    assert first["ok"], first.get("detail") or first.get("reason")
    Path(first["cache_path"]).unlink()
    recovered = accel_bridge.load_library(first)
    assert int(recovered.msim_abi_version()) == 2
    assert Path(first["cache_path"]).is_file()

    second = accel_build.build_accel(force=True)
    assert second["ok"], second.get("detail") or second.get("reason")
    corrupt = Path(second["cache_path"])
    corrupt.write_bytes(b"this is not a shared library")
    retried = accel_bridge.load_library(second)
    assert int(retried.msim_abi_version()) == 2
    # Recovery publishes a new generation; the corrupt inode is left intact.
    assert corrupt.read_bytes() == b"this is not a shared library"
    assert len(accel_build._artifact_paths(accel_build.cache_root())) >= 2


def test_manifest_generation_sorts_numerically_and_ignores_non_object(
        monkeypatch, tmp_path):
    monkeypatch.setenv("MINDIE_CACHE_HOME", str(tmp_path / "cache"))
    root = accel_build.cache_root()
    root.mkdir(parents=True)
    extension = accel_build.library_extension()
    digest = "a" * 64
    older = root / f"{digest}-9{extension}"
    newer = root / f"{digest}-10{extension}"
    for library, generation in ((older, 9), (newer, 10)):
        library.write_bytes(b"native")
        manifest = root / f"{digest}-{generation}{extension}.json"
        manifest.write_text(json.dumps({"digest": digest,
                                        "library": str(library)}))
    malformed = root / f"{digest}-11{extension}.json"
    malformed.write_text("[1]")

    assert accel_build._read_manifest(digest) == newer
    accel_build._prune_engine_cache(root)
    assert not malformed.exists()
    assert older.is_file() and newer.is_file()


def test_prune_sweeps_stale_snapshots_builds_and_trash(tmp_path):
    cache = tmp_path / "engine"
    cache.mkdir()
    extension = accel_build.library_extension()
    old = time.time() - 7200
    stale_src = cache / "src-123-456-789"
    stale_src.mkdir()
    (stale_src / "leftover.c").write_text("/* stale */")
    os.utime(stale_src, (old, old))
    fresh_src = cache / "src-fresh"
    fresh_src.mkdir()
    stale_build = cache / f"build-123-456-789{extension}"
    stale_build.write_bytes(b"stale")
    os.utime(stale_build, (old, old))
    fresh_build = cache / f"build-fresh{extension}"
    fresh_build.write_bytes(b"fresh")
    stale_trash = cache / ".trash-1-2-3-leftover.so.tmp"
    stale_trash.write_bytes(b"stale")
    os.utime(stale_trash, (old, old))
    fresh_trash = cache / ".trash-4-5-6-fresh.so.tmp"
    fresh_trash.write_bytes(b"fresh")

    accel_build._prune_engine_cache(cache)

    assert not stale_src.exists()
    assert not stale_build.exists()
    assert not stale_trash.exists()
    assert fresh_src.is_dir()
    assert fresh_build.is_file()
    assert fresh_trash.is_file()


def test_compiler_identity_contains_full_resolved_path(monkeypatch):
    compiler = _compiler_or_skip()
    monkeypatch.setenv("MINDIE_SIM_CC", compiler)
    monkeypatch.setenv("SDKROOT", "")
    accel_build.reset_build_memo()
    identity = accel_build._compiler_identity(compiler)
    assert identity.split("|", 1)[0] == str(Path(compiler).resolve())


def test_short_root_fixture_uses_missing_platform_temp(monkeypatch):
    """Simulate an absent platform temp parent (Windows-style) and compose."""
    import conftest

    if os.name == "nt":
        pytest.skip("POSIX path used to simulate a missing temp parent")
    missing = Path("/tmp") / f"mprobe-{os.getpid()}"
    assert len(os.fspath(missing).encode("utf-8")) <= (
        conftest.SHORT_PARENT_MAX_BYTES)
    monkeypatch.setattr(conftest.tempfile, "gettempdir", lambda: str(missing))
    saved = os.environ.pop("MINDIE_REPRO_SHORT_ROOT", None)
    patch = pytest.MonkeyPatch()
    try:
        fixture = conftest.isolated_short_root.__wrapped__(patch)
        root = next(fixture)
        assert missing.is_dir()
        assert root.parent == missing
        assert os.environ["MINDIE_REPRO_SHORT_ROOT"] == str(root)
        patch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root / "override"))
        fixture.close()
        assert not root.exists()
        patch.undo()
        assert os.environ.get("MINDIE_REPRO_SHORT_ROOT") is None
    finally:
        patch.undo()
        if saved is not None:
            os.environ["MINDIE_REPRO_SHORT_ROOT"] = saved
        try:
            missing.rmdir()
        except OSError:
            pass


def test_short_root_parent_avoids_a_long_platform_temp(monkeypatch, tmp_path):
    """A long TMPDIR must not silently fall back to the host's shared root."""
    import conftest

    if os.name == "nt":
        pytest.skip("short POSIX fallback is not used on Windows")
    long_parent = tmp_path / ("long-" + "x" * 40)
    monkeypatch.setattr(conftest.tempfile, "gettempdir", lambda: str(long_parent))
    parent = conftest._short_root_parent()
    assert long_parent.is_dir()
    assert parent != long_parent
    assert len(os.fspath(parent).encode("utf-8")) <= (
        conftest.SHORT_PARENT_MAX_BYTES)


def test_temp_cleanup_retains_a_path_until_removal_succeeds(monkeypatch):
    """Transient Windows contention must retry instead of dropping ownership."""
    import temp_cleanup

    root = temp_cleanup.tempdir("retained-")
    saved = temp_cleanup._ROOTS[:]
    temp_cleanup._ROOTS[:] = [root]
    real_rmtree = temp_cleanup.shutil.rmtree

    def blocked(_path, *_args, **_kwargs):
        return None

    try:
        monkeypatch.setattr(temp_cleanup.shutil, "rmtree", blocked)
        temp_cleanup.cleanup()
        assert temp_cleanup._ROOTS == [root]
        monkeypatch.setattr(temp_cleanup.shutil, "rmtree", real_rmtree)
        temp_cleanup.cleanup()
        assert temp_cleanup._ROOTS == []
        assert not root.exists()
    finally:
        temp_cleanup._ROOTS[:] = saved
        real_rmtree(root, ignore_errors=True)


def test_round6_fallback_keeps_a_fake_host_sentinel(monkeypatch):
    """The F10 fallback test must never delete a real host reproducer root."""
    pytest.importorskip("mindie_profiler")
    # The round-6 closure probe is an optional external test module; skip when
    # a standalone checkout does not carry it.
    round6 = pytest.importorskip("test_nr_s6_round6_closure")

    fake_default = round6._fake_mprof_root()
    sentinel = fake_default / "host-sentinel.txt"
    sentinel.write_text("host-owned data")
    monkeypatch.setattr(round6, "_fake_mprof_root", lambda: fake_default)
    round6.test_f10_unicode_short_root_falls_back_to_a_bounded_ascii_receipt(
        monkeypatch)
    assert sentinel.read_text() == "host-owned data"


def test_release_metadata_is_aligned_for_0_9_15():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    version = project["project"]["version"]
    assert version == "0.9.19"
    assert mindie_sim.__version__ == version
    for name in ("sim-0.9.15.md", "sim-0.9.15.en.md",
                 f"sim-{version}.md", f"sim-{version}.en.md"):
        assert (ROOT / "docs" / "releases" / name).is_file()
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8")
    assert f"release-notes-path: docs/releases/sim-{version}.md" in workflow
