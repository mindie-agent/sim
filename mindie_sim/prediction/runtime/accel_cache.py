"""Cross-process locking and retention for the content-addressed engine cache.

[Role] Serialize cache metadata mutation across threads and processes and bound
  the optional C11 engine cache with oldest-first atomic eviction.
[Invariants] Every metadata mutation runs under one advisory lock file inside
  the cache directory.  Published libraries are immutable and are only renamed
  to a ``.trash-*`` sibling before unlink, so a process that mapped the inode
  keeps a valid handle.  Retention never evicts the caller's ``keep_key``;
  stale snapshots, build temporaries and trash files wait out
  ``STALE_TEMP_SECONDS``.  A manifest payload that is not a JSON object is
  treated as absent instead of trusted.
[Timeouts & Locks] ``fcntl.flock`` on POSIX and ``msvcrt.locking`` on Windows,
  attempted non-blocking in a bounded loop.  The process-local re-entrant
  ``_THREAD_LOCK`` lets a locked loader nest ``build_accel``.  A wait timeout
  raises ``TimeoutError`` instead of silently continuing unlocked.
[Side-Effect Policy] Reads, writes and removes only paths inside the given
  cache directory; no model, hardware or network access.
"""
from __future__ import annotations

import contextlib
import errno
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

try:  # POSIX advisory locks
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None
try:  # Windows advisory locks
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

MAX_CACHED_ARTIFACTS = 16
MAX_COMPILER_MARKERS = 16
STALE_TEMP_SECONDS = 3600.0
LOCK_NAME = ".accel_cache.lock"
LOCK_TIMEOUT_SECONDS = 60.0
LOCK_FILE_MODE = 0o600
LOCK_OFFSET = 0
LOCK_BYTES = 1
DEPTH_ZERO = 0
DEPTH_ONE = 1
SINGLE_STEP = 1
MIN_WAIT_SECONDS = 0.0
MISSING_MTIME_NS = 0
DIGEST_HEX_LENGTH = 64
_LOCK_POLL_SECONDS = 0.01
_BUSY_ERRNOS = {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK}

_THREAD_LOCK = threading.RLock()
_LOCK_STATE = threading.local()


def library_extension() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt":
        return ".dll"
    return ".so"


def _depth() -> int:
    return getattr(_LOCK_STATE, "depth", DEPTH_ZERO)


def _try_lock(descriptor: int) -> None:
    if fcntl is not None:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    elif msvcrt is not None:
        os.lseek(descriptor, LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, LOCK_BYTES)
    else:  # pragma: no cover - unsupported platform
        raise OSError(errno.ENOTSUP, "advisory file locking unavailable")


def _release(descriptor: int) -> None:
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        elif msvcrt is not None:
            os.lseek(descriptor, LOCK_OFFSET, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_UNLCK, LOCK_BYTES)
    except TimeoutError:
        raise
    except OSError:
        pass
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _acquire(path: Path, timeout: float) -> int:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT, LOCK_FILE_MODE)
    except TimeoutError:
        raise
    deadline = time.monotonic() + max(MIN_WAIT_SECONDS, float(timeout))
    while True:
        try:
            _try_lock(descriptor)
            return descriptor
        except TimeoutError:
            os.close(descriptor)
            raise
        except OSError as error:
            if error.errno not in _BUSY_ERRNOS:
                os.close(descriptor)
                raise
            if time.monotonic() >= deadline:
                os.close(descriptor)
                raise TimeoutError(
                    f"accel_cache_lock_timeout:{path}") from error
            time.sleep(_LOCK_POLL_SECONDS)


@contextlib.contextmanager
def cache_lock(cache_dir: str | Path, *, timeout: float | None = None):
    """Hold the cache-directory advisory lock (re-entrant per thread).

    The OS lock is taken only at depth zero; a nested enter records another
    level so a locked loader can call :func:`build_accel` without self-deadlock.
    A concurrent process that waits longer than ``timeout`` raises
    ``TimeoutError`` and never proceeds unlocked.
    """
    path = Path(cache_dir) / LOCK_NAME
    wait = LOCK_TIMEOUT_SECONDS if timeout is None else float(timeout)
    wait = max(MIN_WAIT_SECONDS, wait)
    deadline = time.monotonic() + wait
    if not _THREAD_LOCK.acquire(timeout=wait):
        raise TimeoutError(f"accel_cache_lock_timeout:{path}")
    try:
        depth = _depth()
        descriptor = None
        if depth == DEPTH_ZERO:
            remaining = max(MIN_WAIT_SECONDS, deadline - time.monotonic())
            descriptor = _acquire(path, remaining)
        _LOCK_STATE.depth = depth + DEPTH_ONE
        try:
            yield path
        finally:
            _LOCK_STATE.depth = depth
            if descriptor is not None:
                _release(descriptor)
    finally:
        _THREAD_LOCK.release()


def touch_artifact(path: Path) -> None:
    """Best-effort mtime refresh; ``TimeoutError`` is never swallowed."""
    try:
        os.utime(path, None)
    except TimeoutError:
        raise
    except OSError:
        pass


def read_manifest_payload(path: Path) -> dict | None:
    """Parse one manifest object; malformed or non-object JSON is absent."""
    try:
        payload = json.loads(path.read_text())
    except TimeoutError:
        raise
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _mtime_ns(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except TimeoutError:
        raise
    except OSError:
        return MISSING_MTIME_NS


def artifact_paths(cache_dir: Path) -> list[Path]:
    """Content-generation libraries: ``<64-hex digest>-<generation>.<ext>``."""
    extension = library_extension()
    try:
        entries = list(cache_dir.iterdir())
    except TimeoutError:
        raise
    except OSError:
        return []
    artifacts = []
    for path in entries:
        try:
            if not path.is_file():
                continue
        except TimeoutError:
            raise
        except OSError:
            continue
        name = path.name
        if not name.endswith(extension):
            continue
        stem = name[:-len(extension)]
        digest, separator, generation = stem.partition("-")
        if (separator and len(digest) == DIGEST_HEX_LENGTH
                and all(char in "0123456789abcdef" for char in digest)
                and generation.isdigit()):
            artifacts.append(path)
    return artifacts


def _discard_path(path: Path) -> bool:
    """Atomically quarantine then unlink; mapped Windows files are deferred.

    Once ``os.replace(path, trash)`` succeeds, ``path`` is already vacated and
    a later unlink failure leaves only a ``.trash-*`` sibling.  That sibling is
    not a cache artifact and is reclaimed by :func:`prune_engine_cache` after
    ``STALE_TEMP_SECONDS`` so a transient Windows share violation cannot leak
    it forever.
    """
    trash = path.with_name(
        f".trash-{os.getpid()}-{threading.get_ident()}-"
        f"{time.monotonic_ns()}-{path.name}.tmp")
    try:
        os.replace(path, trash)
    except TimeoutError:
        raise
    except PermissionError:
        return False
    except OSError:
        return False
    try:
        trash.unlink()
    except TimeoutError:
        raise
    except PermissionError:
        pass
    except OSError:
        pass
    return True


def _keep_names(artifacts: list[Path], keep_key: str | Path | None) -> set[str]:
    if keep_key is None:
        return set()
    if isinstance(keep_key, Path):
        return {keep_key.name}
    text = str(keep_key)
    if any(path.name == text for path in artifacts):
        return {text}
    if len(text) == DIGEST_HEX_LENGTH and all(char in "0123456789abcdef" for char in text):
        matches = [path for path in artifacts
                   if path.name.startswith(text + "-")]
        if matches:
            return {max(matches, key=_mtime_ns).name}
    return {Path(text).name}


def _sweep_stale_temporaries(cache_dir: Path, stale_before: float,
                             removed: list[Path]) -> None:
    """Reclaim old build libraries, trash files and source snapshots."""
    candidates = set(cache_dir.glob("*.tmp"))
    candidates.update(cache_dir.glob("build-*"))
    candidates.update(cache_dir.glob(".trash-*"))
    for temporary in sorted(candidates):
        try:
            if not temporary.is_file():
                continue
            stale = temporary.stat().st_mtime < stale_before
        except TimeoutError:
            raise
        except OSError:
            continue
        if stale and _discard_path(temporary):
            removed.append(temporary)
    for directory in sorted(cache_dir.glob("src-*")):
        try:
            if directory.is_symlink() or not directory.is_dir():
                continue
            stale = directory.stat().st_mtime < stale_before
        except TimeoutError:
            raise
        except OSError:
            continue
        if not stale:
            continue
        try:
            shutil.rmtree(directory)
        except TimeoutError:
            raise
        except OSError:
            continue
        removed.append(directory)


def prune_engine_cache(cache_dir: Path, *, keep_key: str | Path | None = None,
                       max_artifacts: int = MAX_CACHED_ARTIFACTS) -> list[Path]:
    """Bound engine libraries and manifests by oldest-first atomic eviction.

    The newest ``max_artifacts`` generation files are retained and the current
    ``keep_key`` (digest, filename or path) is never evicted.  When an older
    Windows process still maps a ``.dll``/``.pyd``, rename or unlink can fail;
    that eviction is deferred and its manifest stays valid for a later run.
    Stale compiler markers beyond 16 entries and temporaries older than one
    hour are reclaimed independently of artifact retention.
    """
    removed: list[Path] = []
    artifacts = artifact_paths(cache_dir)
    keep = _keep_names(artifacts, keep_key)
    ordered = sorted(artifacts, key=lambda path: (_mtime_ns(path), path.name))
    remaining = len(ordered)
    limit = max(DEPTH_ZERO, int(max_artifacts))
    for path in ordered:
        if remaining <= limit:
            break
        if path.name in keep:
            continue
        if not _discard_path(path):
            continue
        removed.append(path)
        remaining -= SINGLE_STEP
        manifest = path.with_suffix(path.suffix + ".json")
        if _discard_path(manifest):
            removed.append(manifest)
    for manifest in sorted(cache_dir.glob("*.json")):
        payload = read_manifest_payload(manifest)
        library = Path(str((payload or {}).get("library", "")))
        try:
            orphan = not library.is_file()
        except TimeoutError:
            raise
        except OSError:
            orphan = True
        if orphan and _discard_path(manifest):
            removed.append(manifest)
    markers = sorted(cache_dir.glob("compiler-*.id"),
                     key=_mtime_ns, reverse=True)
    for marker in markers[max(DEPTH_ZERO, int(MAX_COMPILER_MARKERS)):]:
        if _discard_path(marker):
            removed.append(marker)
    _sweep_stale_temporaries(
        cache_dir, time.time() - STALE_TEMP_SECONDS, removed)
    return removed
