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
import tempfile
import time
import tomllib

import pytest

import mindie_sim
from mindie_sim.prediction.runtime import accel_build

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
    assert list(cache.iterdir()) == []


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


def test_issue34_budget_removes_internal_artifacts(tmp_path, monkeypatch):
    module = _script("check_issue34_test_budget")
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    code = module.main([
        "--skip-tests", "--root", str(ROOT),
        "--pattern", "tests/no-such-issue34-file-*.py",
    ])
    assert code == 0
    assert list(scratch.iterdir()) == []


def test_release_metadata_is_aligned_for_0_9_15():
    assert mindie_sim.__version__ == "0.9.15"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["version"] == "0.9.15"
    for name in ("sim-0.9.15.md", "sim-0.9.15.en.md"):
        assert (ROOT / "docs" / "releases" / name).is_file()
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8")
    assert "release-notes-path: docs/releases/sim-0.9.15.md" in workflow
