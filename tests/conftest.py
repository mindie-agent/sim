"""Shared isolated-runtime and isolated-cache fixtures for MindIE-Sim tests.

[Role] Keep user HOME/config/data/diagnostics/cache directories and accelerator
  build memos untouched across every test module.
[Invariants] `isolated_cache` provides one session-local `MINDIE_CACHE_HOME` and
  cleans it on teardown; `isolated_short_root` gives reproducer storage a fresh
  short ASCII root under the platform temp parent (or the short POSIX temp when
  that path is too long for the product's 48-byte contract) so concurrent runs
  cannot collide in the shared host reproducer directory;
  `isolated_runtime` isolates per-test runtime paths. Every session temporary
  root is also tracked for session/process cleanup.
[Timeouts & Locks] No threads, sleeps, or external I/O beyond temporary dirs.
[Side-Effect Policy] Writes and removes only temporary directories created by
  pytest/tempfile.
"""
from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = Path(__file__).resolve().parent
#: ``repro_storage`` ignores roots longer than 48 bytes; leave room for
#: ``mprof-`` plus the 8-character ``mkdtemp`` suffix.
SHORT_PARENT_MAX_BYTES = 34
for _entry in (ROOT, TESTS):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from temp_cleanup import cleanup as cleanup_tempdirs  # noqa: E402
from temp_cleanup import tempdir  # noqa: E402
from mindie_sim.observations.kernels import PROFILER_API_REQUIREMENT  # noqa: E402
from mindie_sim.prediction.runtime import accel_build  # noqa: E402


def pytest_sessionfinish(session, exitstatus):
    """Remove tracked temporary roots even if a fixture teardown was skipped."""
    cleanup_tempdirs()


@pytest.fixture(scope="session", autouse=True)
def isolated_cache():
    base = tempdir("nr-s3-accel-")
    names = (
        "MINDIE_CACHE_HOME",
        "MINDIE_DATA_HOME",
        "XDG_CACHE_HOME",
        "MINDIE_SIM_ACCEL",
        "MINDIE_SIM_ENGINE_BACKEND",
        "MINDIE_SIM_CC",
    )
    old = {name: os.environ.get(name) for name in names}
    os.environ["MINDIE_CACHE_HOME"] = str(base)
    for name in names[1:]:
        os.environ.pop(name, None)
    accel_build.reset_build_memo()
    try:
        yield Path(base)
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        accel_build.reset_build_memo()
        shutil.rmtree(base, ignore_errors=True)


def _short_root_parent() -> Path:
    """Platform temp parent, with a short POSIX fallback when it is too long.

    ``repro_storage.short_root`` ignores ``MINDIE_REPRO_SHORT_ROOT`` beyond 48
    bytes.  ``TMPDIR`` on macOS and some CI hosts is much longer, and a long
    root would silently send tests to the shared host reproducer directory.
    Prefer the platform temp directory when it fits and use the platform's
    short POSIX temp directory only when necessary.
    """
    parent = Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    if len(os.fspath(parent).encode("utf-8")) <= SHORT_PARENT_MAX_BYTES:
        return parent
    if os.name != "nt":
        fallback = Path("/tmp")
        try:
            fallback.mkdir(parents=True, exist_ok=True)
        except OSError:
            return parent
        if len(os.fspath(fallback).encode("utf-8")) <= SHORT_PARENT_MAX_BYTES:
            return fallback
    return parent


@pytest.fixture(autouse=True)
def isolated_short_root(monkeypatch):
    """Own a fresh short root under the platform temp parent.

    ``monkeypatch`` owns the environment write so it unwinds in the same LIFO
    order as any test-level override, never restoring a deleted directory.
    """
    root = tempdir("mprof-", directory=_short_root_parent())
    monkeypatch.setenv("MINDIE_REPRO_SHORT_ROOT", str(root))
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session")
def _worker_runtime_root():
    root = tempdir("nr-s3-runtime-").resolve()
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolated_runtime(_worker_runtime_root, monkeypatch):
    with os.scandir(_worker_runtime_root) as it:
        dirty = [entry.path for entry in it]
    for entry_path in dirty:
        if os.path.isdir(entry_path) and not os.path.islink(entry_path):
            shutil.rmtree(entry_path, ignore_errors=True)
        else:
            os.unlink(entry_path)
    monkeypatch.setenv("HOME", str(_worker_runtime_root / "home"))
    monkeypatch.setenv("MINDIE_CONFIG_HOME", str(_worker_runtime_root / "config"))
    monkeypatch.setenv("MINDIE_DATA_HOME", str(_worker_runtime_root / "data"))
    monkeypatch.setenv("MINDIE_DIAGNOSTICS_ROOT", str(_worker_runtime_root / "diagnostics"))


@pytest.fixture
def profiler_api():
    for module in ("mindie_profiler.normalized", "mindie_profiler.tensor_contracts"):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name not in {"mindie_profiler", module}:
                raise
            pytest.skip(PROFILER_API_REQUIREMENT)
