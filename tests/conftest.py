"""Root fixtures: zero host pollution, isolated accelerator cache, requires_* markers.

[Role] Keep every user/host directory (HOME, APPDATA, XDG_*, MINDIE_*, auth,
  uv cache, TMPDIR) and accelerator build memos untouched by any test, and turn
  optional sibling/mount requirements into explicit markers.
[Invariants] All 15 host-path variables point into pytest-owned temporary
  roots for every test; ``MINDIE_CACHE_HOME`` is one session-local root so the
  C11 accelerator builds once per worker; ``isolated_short_root`` gives
  reproducer storage a fresh short ASCII root (<= 48 bytes contract);
  ``requires_profiler`` / ``requires_capture_mount`` skip with a visible
  reason locally and FAIL when ``MINDIE_SIM_REQUIRE_ALL=1`` (CI), so missing
  coverage is never silent.
[Timeouts & Locks] No threads, sleeps, or external I/O beyond temporary dirs.
[Side-Effect Policy] Writes and removes only temporary directories created by
  pytest/tempfile; environment changes are owned by ``MonkeyPatch`` and unwind.
"""
from __future__ import annotations

import importlib
import json
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
REQUIRE_ALL_ENV = "MINDIE_SIM_REQUIRE_ALL"
CAMPAIGN_PATHS_ENV = "MINDIE_SIM_CAMPAIGN_PATHS"
DEFAULT_CAMPAIGNS_DIR = ROOT / "data" / "verification" / "campaigns"
GOLDEN_CAMPAIGN_FIXTURE = TESTS / "fixtures" / "bench_matrix" / "v2_contract_golden.json"
VA_COMMIT_SHORT_LEN = 8
#: Per-test host-path variables -> subdirectory of the worker runtime root.
HOST_PATH_VARS = {
    "HOME": "home",
    "USERPROFILE": "home",
    "APPDATA": "appdata/roaming",
    "LOCALAPPDATA": "appdata/local",
    "XDG_CONFIG_HOME": "xdg/config",
    "XDG_STATE_HOME": "xdg/state",
    "XDG_DATA_HOME": "xdg/data",
    "MINDIE_CONFIG_HOME": "config",
    "MINDIE_DATA_HOME": "data",
    "MINDIE_DIAGNOSTICS_ROOT": "diagnostics",
    "REMOTE_DEV_AUTH_DIR": "remote-dev-auth",
    "UV_CACHE_DIR": "uv-cache",
}
#: Session-scoped variables ``MINDIE_CACHE_HOME``, ``XDG_CACHE_HOME`` and
#: ``TMPDIR`` live in ``isolated_cache`` (the accelerator cache must survive).
ACCEL_OVERRIDE_VARS = ("MINDIE_SIM_ACCEL", "MINDIE_SIM_ENGINE_BACKEND", "MINDIE_SIM_CC")
for _entry in (ROOT, TESTS):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from _support.sharding import partition_shard_items  # noqa: E402
from _support.tempdirs import cleanup as cleanup_tempdirs  # noqa: E402
from _support.tempdirs import tempdir  # noqa: E402
from mindie_sim.observations.captures import PROFILER_API_REQUIREMENT  # noqa: E402
from mindie_sim.prediction.runtime import accel_build  # noqa: E402


def pytest_addoption(parser):
    group = parser.getgroup("sharding")
    group.addoption(
        "--shard-index", action="store", type=int, default=0,
        help="0-based shard index for deterministic CI sharding (default: 0)",
    )
    group.addoption(
        "--num-shards", action="store", type=int, default=1,
        help="Total number of deterministic CI shards (default: 1)",
    )
    camp = parser.getgroup("campaign")
    camp.addoption(
        "--campaign-manifest", action="append", default=[],
        help="Path(s) to mindie-bench-campaign/1 JSON manifest(s) to reconcile",
    )
    camp.addoption(
        "--campaign-dir", action="append", default=[],
        help="Directory(ies) recursively searched for campaign_manifest.json / *.campaign.json",
    )
    camp.addoption(
        "--campaign-sku", action="store", default=None,
        help="Optional SKU filter for dynamically mounted campaigns",
    )
    camp.addoption(
        "--campaign-cann-version", action="store", default=None,
        help="Optional CANN version filter for dynamically mounted campaigns",
    )
    camp.addoption(
        "--campaign-va-commit", action="store", default=None,
        help="Optional vllm-ascend commit prefix filter for dynamically mounted campaigns",
    )


def _discover_campaign_manifests(config) -> tuple[list[tuple[Path, str]], bool]:
    """Discover dynamically mounted campaign manifests from CLI flags, env, or default dir."""
    raw_manifests = list(config.getoption("--campaign-manifest") or [])
    raw_dirs = list(config.getoption("--campaign-dir") or [])
    env_raw = os.environ.get(CAMPAIGN_PATHS_ENV, "").strip()
    env_entries = [p.strip() for p in env_raw.split(os.pathsep) if p.strip()] if env_raw else []
    sku_filter = config.getoption("--campaign-sku")
    cann_filter = config.getoption("--campaign-cann-version")
    va_filter = config.getoption("--campaign-va-commit")
    has_explicit_source = bool(raw_manifests or raw_dirs or env_entries)
    explicit = has_explicit_source or bool(sku_filter or cann_filter or va_filter)

    candidates: list[Path] = []
    for raw in raw_manifests:
        candidates.append(Path(raw))
    for raw in raw_dirs:
        base = Path(raw)
        if base.is_dir():
            candidates.extend(sorted(base.rglob("campaign_manifest.json")))
            candidates.extend(sorted(base.rglob("*.campaign.json")))
    for raw in env_entries:
        p = Path(raw)
        if p.is_file():
            candidates.append(p)
        elif p.is_dir():
            candidates.extend(sorted(p.rglob("campaign_manifest.json")))
            candidates.extend(sorted(p.rglob("*.campaign.json")))
    if not has_explicit_source and DEFAULT_CAMPAIGNS_DIR.is_dir():
        candidates.extend(sorted(DEFAULT_CAMPAIGNS_DIR.rglob("campaign_manifest.json")))
        candidates.extend(sorted(DEFAULT_CAMPAIGNS_DIR.rglob("*.campaign.json")))

    seen: set[Path] = set()
    entries: list[tuple[Path, str]] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        try:
            doc = json.loads(resolved.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            entries.append((resolved, resolved.name))
            continue
        if not isinstance(doc, dict):
            continue
        fp = doc.get("version_fingerprint") if isinstance(doc.get("version_fingerprint"), dict) else {}
        sku = str(fp.get("sku") or "unknown")
        cann = str(fp.get("cann_version") or "unknown")
        va = str(fp.get("va_commit") or "unknown")
        cid = str(doc.get("campaign_id") or resolved.stem)
        if sku_filter and sku != sku_filter:
            continue
        if cann_filter and cann != cann_filter:
            continue
        if va_filter and not va.startswith(va_filter):
            continue
        test_id = f"{sku}[cann={cann},va={va[:VA_COMMIT_SHORT_LEN]}]::{cid}"
        entries.append((resolved, test_id))
    return entries, explicit


def pytest_generate_tests(metafunc):
    if "mounted_campaign_manifest" not in metafunc.fixturenames:
        return
    entries, explicit = _discover_campaign_manifests(metafunc.config)
    if entries:
        metafunc.parametrize(
            "mounted_campaign_manifest",
            [path for path, _ in entries],
            ids=[label for _, label in entries],
        )
    elif explicit:
        metafunc.parametrize(
            "mounted_campaign_manifest",
            [pytest.param(None, id="explicit-campaign-mount-empty")],
        )
    else:
        metafunc.parametrize(
            "mounted_campaign_manifest",
            [pytest.param(GOLDEN_CAMPAIGN_FIXTURE, id="golden-v2-contract-campaign")],
        )


def pytest_collection_modifyitems(config, items):
    num_shards = config.getoption("--num-shards")
    shard_index = config.getoption("--shard-index")
    try:
        selected, deselected = partition_shard_items(items, shard_index, num_shards)
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    if deselected:
        items[:] = selected
        config.hook.pytest_deselected(items=deselected)


def _missing(reason: str) -> None:
    """Skip visibly in local runs; fail in CI so missing coverage is never silent."""
    if os.environ.get(REQUIRE_ALL_ENV) == "1":
        pytest.fail(f"{reason} ({REQUIRE_ALL_ENV}=1)", pytrace=False)
    pytest.skip(reason)


def _profiler_available() -> bool:
    for module in ("mindie_profiler.normalized", "mindie_profiler.tensor_contracts"):
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            if exc.name not in {"mindie_profiler", module}:
                raise
            return False
    return True


def pytest_runtest_setup(item):
    if item.get_closest_marker("requires_profiler") and not _profiler_available():
        _missing(PROFILER_API_REQUIREMENT)
    absent: list[str] = []
    for mount in item.iter_markers("requires_capture_mount"):
        for raw in mount.args:
            target = Path(raw) if Path(raw).is_absolute() else ROOT / raw
            if not target.exists() and str(raw) not in absent:
                absent.append(str(raw))
    if absent:
        _missing(f"retained capture unavailable: {', '.join(absent)}")


def pytest_sessionfinish(session, exitstatus):
    """Remove tracked temporary roots even if a fixture teardown was skipped."""
    cleanup_tempdirs()


@pytest.fixture(scope="session", autouse=True)
def isolated_cache():
    """Session-level cache, temp dir, and host-path defaults (protects module fixtures)."""
    old_tempdir = tempfile.tempdir
    base = tempdir("msim-session-")
    session_tmp = base / "tmp"
    session_tmp.mkdir()
    session_home = base / "session"
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("MINDIE_CACHE_HOME", str(base / "cache"))
            patch.setenv("XDG_CACHE_HOME", str(base / "xdg-cache"))
            for temp_var in ("TMPDIR", "TEMP", "TMP"):
                patch.setenv(temp_var, str(session_tmp))
            patch.setattr(tempfile, "tempdir", str(session_tmp))
            for name, sub in HOST_PATH_VARS.items():
                patch.setenv(name, str(session_home / sub))
            for name in ACCEL_OVERRIDE_VARS:
                patch.delenv(name, raising=False)
            accel_build.reset_build_memo()
            try:
                yield base / "cache"
            finally:
                accel_build.reset_build_memo()
    finally:
        tempfile.tempdir = old_tempdir
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
    root = tempdir("msim-runtime-").resolve()
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(autouse=True)
def isolated_runtime(_worker_runtime_root, monkeypatch):
    """Point every per-test host-path variable at a fresh directory tree."""
    with os.scandir(_worker_runtime_root) as it:
        dirty = [entry.path for entry in it]
    for entry_path in dirty:
        if os.path.isdir(entry_path) and not os.path.islink(entry_path):
            shutil.rmtree(entry_path, ignore_errors=True)
        else:
            os.unlink(entry_path)
    for name, sub in HOST_PATH_VARS.items():
        monkeypatch.setenv(name, str(_worker_runtime_root / sub))


@pytest.fixture
def profiler_api():
    if not _profiler_available():
        _missing(PROFILER_API_REQUIREMENT)


@pytest.fixture
def lean_binary() -> str:
    elan = Path("/tmp/elan-mindie-s3b/bin/lean")
    lean = shutil.which("lean") or (str(elan) if elan.is_file() else None)
    if not lean:
        _missing("Lean toolchain unavailable")
    assert lean is not None
    return lean
