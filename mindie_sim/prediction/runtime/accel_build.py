"""Client-side build/cache lifecycle for the optional C11 engine kernel.

[Role] Discover a real local C compiler, compile the shipped pure C11 sources
  with exact semantic flags, validate the produced library in a bounded
  subprocess, and publish it under an immutable content+generation address.
[Invariants] The wheel ships source only: no build-time native extension, no
  compiler download and no network.  The cache key hashes every shipped source
  byte, the resolved compiler binary/version, platform/pointer-width/endianness
  identity and every semantic flag; the caller's raw ``PATH`` is not an input,
  so every virtualenv resolving the same system compiler shares one artifact.
  A validated library path is never overwritten: each build publishes a fresh
  generation file plus an atomic readiness manifest.  Temporary names are unique
  per process, thread and monotonic tick, and failed publications remove their
  snapshot, temporary and candidate files.  A build failure never tombstones
  the key and never changes simulation semantics.
[Timeouts & Locks] One argv-list subprocess with no shell; compile budget 20 s,
  compiler probe 15 s, ABI validation 15 s.  Cache inspection, hit refresh,
  publication and retention share one cross-process advisory lock file with a
  bounded wait; a hit refreshes its mtime under that lock before pruning, and
  the loader keeps the lock across ``ctypes.CDLL``.  Publication is an atomic
  ``os.link`` into a fresh generation path; an existing library inode is never
  replaced.  Retention is bounded by ``MAX_CACHED_ARTIFACTS`` and evicts
  oldest-first via atomic rename; a mapped Windows library defers eviction
  without failing.
[Side-Effect Policy] Reads the packaged sources and environment; writes and
  prunes only the content-addressed cache directory.  No model, hardware or
  network access.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from . import accel_cache
from .accel_cache import MAX_CACHED_ARTIFACTS, library_extension
from .accel_tables import PROBE_GOLDEN

SOURCE_NAMES = (
    "_engine_accel.c",
    "_engine_accel_abi.h",
    "_engine_accel_core.h",
    "_engine_accel_rates.h",
    "_engine_accel_submit.h",
    "_engine_accel_terms.h",
)
CACHE_SUBDIR = ("mindie", "sim", "engine")
COMPILE_TIMEOUT_SECONDS = 20.0
#: Cold clang/dyld launches can exceed 2 s under load; the probe is bounded but
#: must not flake into ``abi_probe_timeout`` on a busy host.
PROBE_TIMEOUT_SECONDS = 15.0
VALIDATE_TIMEOUT_SECONDS = 15.0
MAX_DIAGNOSTIC_BYTES = 2000
IDENTITY_MARKER_HEX = 16
INVALID_GENERATION = -1
GENERATION_PART = 0
_BASE_FLAGS = (
    "-std=c11", "-O3", "-fPIC", "-fno-fast-math", "-ffp-contract=off",
    "-fno-strict-aliasing",
)
_STATUS_MEMO: dict[tuple, dict] = {}
_IDENTITY_MEMO: dict[tuple, str] = {}
_MEMO_LOCK = threading.Lock()
_SUBPROCESS_KWARGS = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
#: Retention and locking live in the small coordinator module; these aliases
#: preserve the established private API used by cache tests and diagnostics.
MAX_COMPILER_MARKERS = accel_cache.MAX_COMPILER_MARKERS
STALE_TEMP_SECONDS = accel_cache.STALE_TEMP_SECONDS
_artifact_paths = accel_cache.artifact_paths
_mtime_ns = accel_cache._mtime_ns
_keep_names = accel_cache._keep_names
_discard_path = accel_cache._discard_path
_prune_engine_cache = accel_cache.prune_engine_cache


def source_dir() -> Path:
    return Path(__file__).parent


def source_path() -> Path:
    return source_dir() / SOURCE_NAMES[0]


def source_snapshot() -> tuple[bytes, dict[str, bytes]]:
    """Exact bytes of every shipped source plus the documented digest."""
    digest = hashlib.sha256()
    files: dict[str, bytes] = {}
    for name in SOURCE_NAMES:
        data = (source_dir() / name).read_bytes()
        files[name] = data
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(name.encode())
        digest.update(data)
    return digest.digest(), files


def source_bytes() -> bytes:
    """Digest over every shipped source byte (the cache-key preimage)."""
    return source_snapshot()[0]


def compiler_flags() -> list[str]:
    """Exact deterministic flags; never ``-march=native`` or fast-math."""
    flags = [f for f in _BASE_FLAGS if not (os.name == "nt" and f == "-fPIC")]
    if sys.platform == "darwin":
        flags.extend(("-dynamiclib", "-lm"))
    elif os.name == "nt":
        flags.append("-shared")
    else:
        flags.extend(("-shared", "-lm"))
    return flags


def cache_root() -> Path:
    """``MINDIE_CACHE_HOME`` -> ``MINDIE_DATA_HOME/cache`` -> ``XDG_CACHE_HOME``
    -> ``~/.cache``, under the documented ``mindie/sim/engine`` subpath."""
    base = os.environ.get("MINDIE_CACHE_HOME")
    if not base:
        data = os.environ.get("MINDIE_DATA_HOME")
        if data:
            base = str(Path(data).expanduser() / "cache")
    if not base:
        base = os.environ.get("XDG_CACHE_HOME")
    if not base:
        base = str(Path.home() / ".cache")
    return Path(base).expanduser().joinpath(*CACHE_SUBDIR)


def discover_compiler() -> tuple[str | None, str | None]:
    """Return ``(absolute_compiler, reason)`` using an explicit override first."""
    explicit = os.environ.get("MINDIE_SIM_CC")
    if explicit:
        found = shutil.which(explicit)
        if found is None:
            return None, "compiler_unavailable:configured_override_missing"
        return found, None
    candidates = ("gcc", "clang", "cc") if os.name == "nt" else ("cc", "clang", "gcc")
    for candidate in candidates:
        found = shutil.which(candidate)
        if found is not None:
            return found, None
    return None, "compiler_unavailable:no_c_toolchain"


def _unique_temp(directory: Path, prefix: str, suffix: str) -> Path:
    handle, name = tempfile.mkstemp(
        prefix=f"{prefix}-{os.getpid()}-{threading.get_ident()}-{time.monotonic_ns()}-",
        suffix=suffix, dir=str(directory))
    os.close(handle)
    return Path(name)


def _resolve_compiler(compiler: str) -> str:
    """Resolve a compiler name to the executable that PATH actually selects."""
    found = shutil.which(compiler) or compiler
    try:
        return str(Path(found).resolve())
    except TimeoutError:
        raise
    except OSError:
        return os.path.abspath(found)


def _compiler_context(compiler: str) -> tuple[str, str, str]:
    """Resolved binary, Stat footprint and SDK root for a toolchain identity.

    The caller's ``PATH`` is deliberately absent: distinct virtualenvs or
    benchmark slots that resolve the same system compiler must share one cache
    key.  Only the selected executable and explicit SDK root are inputs.
    """
    resolved = _resolve_compiler(compiler)
    try:
        stat = os.stat(resolved)
        footprint = f"{stat.st_size}:{stat.st_mtime_ns}"
    except TimeoutError:
        raise
    except OSError:
        footprint = "unknown"
    sdkroot = os.environ.get("SDKROOT", "")
    if sdkroot:
        sdkroot = os.path.normpath(os.path.abspath(
            os.path.expanduser(sdkroot)))
    return resolved, footprint, sdkroot


def _compiler_identity(compiler: str, *, persist: bool = False) -> str:
    resolved, footprint, sdkroot = _compiler_context(compiler)
    platform_identity = _platform_identity()
    key = (resolved, footprint, sdkroot, platform_identity)
    with _MEMO_LOCK:
        cached = _IDENTITY_MEMO.get(key)
    if cached is not None:
        return cached
    marker_key = hashlib.sha256(
        "|".join(("compiler-identity-v3", *key)).encode()).hexdigest()[:IDENTITY_MARKER_HEX]
    marker = cache_root() / f"compiler-{marker_key}.id"
    if persist:
        try:
            stored = marker.read_text().strip()
            if stored:
                with _MEMO_LOCK:
                    _IDENTITY_MEMO[key] = stored
                return stored
        except TimeoutError:
            raise
        except OSError:
            pass
    version = "unknown"
    try:
        completed = subprocess.run(
            [resolved, "--version"], capture_output=True, text=True,
            timeout=PROBE_TIMEOUT_SECONDS, check=False, **_SUBPROCESS_KWARGS)
        lines = (completed.stdout or completed.stderr or "").splitlines()
        if lines:
            version = lines[0].strip()[:160]
    except subprocess.TimeoutExpired:
        version = "timeout"
    except TimeoutError:
        version = "timeout"
    except OSError:
        version = "unavailable"
    identity = (
        f"{resolved}|{footprint}|{version}|{sdkroot}|{platform_identity}")
    with _MEMO_LOCK:
        _IDENTITY_MEMO[key] = identity
    if persist:
        temporary = None
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            temporary = _unique_temp(marker.parent, "identity", ".tmp")
            temporary.write_text(identity)
            os.replace(temporary, marker)
            temporary = None
        except TimeoutError:
            raise
        except OSError:
            pass
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except TimeoutError:
                    raise
                except OSError:
                    pass
    return identity


def _platform_identity() -> str:
    return "|".join((
        sys.platform, platform.machine(), str(sys.getsizeof(0)),
        sys.byteorder, library_extension(),
    ))


def cache_digest(source: bytes, compiler: str, flags: list[str],
                 identity: str | None = None) -> str:
    identity = identity or _compiler_identity(compiler)
    digest = hashlib.sha256()
    for part in (source, identity.encode(), " ".join(flags).encode(),
                 _platform_identity().encode()):
        digest.update(len(part).to_bytes(8, "little"))
        digest.update(part)
    return digest.hexdigest()


def _bounded_detail(text: str | bytes | None) -> str:
    if not text:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    return text[-MAX_DIAGNOSTIC_BYTES:]


def _failure(reason: str, *, compiler=None, identity=None, flags=None,
             digest=None, path=None, detail="", check_only=False) -> dict:
    return {
        "ok": False, "backend": "python_fallback", "reason": reason,
        "detail": _bounded_detail(detail), "compiler": compiler,
        "compiler_identity": identity, "flags": list(flags or ()),
        "digest": digest, "cache_path": None if path is None else str(path),
        "cache_hit": False, "check_only": check_only, "abi": 2,
        "toolchain_ready": compiler is not None, "library_ready": False,
    }


_VALIDATE_BODY = (
    "import ctypes,json,sys\n"
    "lib=ctypes.CDLL(sys.argv[1])\n"
    "lib.msim_abi_version.restype=ctypes.c_uint32\n"
    "probe=(ctypes.c_uint64*16)()\n"
    "lib.msim_abi_probe.restype=ctypes.c_int32\n"
    "lib.msim_abi_probe.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t]\n"
    "rc=int(lib.msim_abi_probe(probe,16))\n"
    "print(json.dumps({'v':int(lib.msim_abi_version()),'rc':rc,"
    "'cells':[int(x) for x in probe]}))\n"
)


def _validate(path: Path) -> tuple[bool, str]:
    """Bounded out-of-process ABI/load validation before any in-process CDLL."""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _VALIDATE_BODY, str(path)],
            capture_output=True, text=True,
            timeout=VALIDATE_TIMEOUT_SECONDS, check=False, **_SUBPROCESS_KWARGS)
    except subprocess.TimeoutExpired:
        return False, "abi_probe_timeout"
    except TimeoutError:
        return False, "abi_probe_timeout"
    except OSError as error:
        return False, f"abi_probe_failed:{error}"
    if completed.returncode != 0:
        return False, f"abi_probe_failed:{_bounded_detail(completed.stderr)}"
    try:
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as error:
        return False, f"abi_probe_invalid:{error}"
    if payload.get("v") != 2 or payload.get("rc") != 0:
        return False, "abi_version_mismatch"
    try:
        cells = tuple(int(x) for x in payload.get("cells", ()))
    except (TypeError, ValueError):
        return False, "abi_probe_invalid"
    if len(cells) != len(PROBE_GOLDEN) or cells != PROBE_GOLDEN:
        return False, "abi_probe_mismatch"
    return True, ""


def _manifest_path(destination: Path) -> Path:
    return destination.with_suffix(destination.suffix + ".json")


def _manifest_generation(candidate: Path, digest: str) -> int:
    """Numeric generation from ``<digest>-<generation>.<ext>.json``."""
    prefix = digest + "-"
    if not candidate.name.startswith(prefix):
        return INVALID_GENERATION
    text = candidate.name[len(prefix):].partition(".")[GENERATION_PART]
    try:
        return int(text)
    except ValueError:
        return INVALID_GENERATION


def _read_manifest(digest: str) -> Path | None:
    """Newest valid generation for ``digest``; malformed JSON is ignored.

    Sorting is numeric (generation 10 ahead of 9), never lexicographic, and a
    non-object payload such as ``[1]`` is treated as absent.
    """
    root = cache_root()
    candidates = sorted(root.glob(f"{digest}-*.json"),
                        key=lambda item: (_manifest_generation(item, digest),
                                          item.name), reverse=True)
    for candidate in candidates:
        payload = accel_cache.read_manifest_payload(candidate)
        if payload is None:
            continue
        path = Path(str(payload.get("library", "")))
        if payload.get("digest") == digest and path.is_file():
            return path
    return None


def _touch_hit(path: Path) -> None:
    """Refresh a hit before pruning so it is never the oldest artifact."""
    accel_cache.touch_artifact(path)


def _publish(compiler: str, flags: list[str], destination_dir: Path,
             files: dict[str, bytes], digest: str) -> tuple[bool, str, Path | None]:
    """Compile an immutable snapshot and publish it under a fresh generation."""
    snapshot_dir = None
    temporary = None
    manifest_temp = None
    candidate = None
    published = False
    try:
        destination_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir = Path(tempfile.mkdtemp(
            prefix=f"src-{os.getpid()}-{threading.get_ident()}-",
            dir=str(destination_dir)))
    except TimeoutError:
        raise
    except OSError as error:
        return False, f"cache_directory_unwritable:{error}", None
    try:
        for name, data in files.items():
            try:
                (snapshot_dir / name).write_bytes(data)
            except TimeoutError:
                raise
            except OSError as error:
                return False, f"cache_directory_unwritable:{error}", None
        snapshot = snapshot_dir / SOURCE_NAMES[0]
        try:
            temporary = _unique_temp(destination_dir, "build", library_extension())
        except TimeoutError:
            raise
        except OSError as error:
            return False, f"cache_directory_unwritable:{error}", None
        command = [compiler, *flags, "-o", str(temporary), str(snapshot)]
        try:
            completed = subprocess.run(
                command, capture_output=True, text=True,
                timeout=COMPILE_TIMEOUT_SECONDS, check=False, **_SUBPROCESS_KWARGS)
        except subprocess.TimeoutExpired:
            return False, "accel_compile_timeout", None
        except TimeoutError:
            return False, "accel_compile_timeout", None
        except OSError as error:
            return False, f"compiler_exec_failed:{error}", None
        if completed.returncode != 0:
            return False, _bounded_detail(completed.stderr or completed.stdout), None
        valid, reason = _validate(temporary)
        if not valid:
            return False, reason, None
        # Publication and retention are one metadata mutation: hold the
        # cross-process lock so a concurrent hit cannot observe a library that
        # is about to be pruned and another builder cannot race the manifest.
        with accel_cache.cache_lock(destination_dir):
            for generation in range(100001):
                target = destination_dir / (
                    f"{digest.hex() if isinstance(digest, bytes) else digest}"
                    f"-{generation}{library_extension()}")
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    continue
                except TimeoutError:
                    raise
                except OSError as error:
                    return False, f"publication_failed:{error}", None
                candidate = target
                break
            if candidate is None:
                return False, "cache_generation_exhausted", None
            try:
                os.unlink(temporary)
                temporary = None
            except TimeoutError:
                raise
            except OSError as error:
                return False, f"publication_failed:{error}", None
            try:
                manifest = _manifest_path(candidate)
                payload = json.dumps({"digest": digest, "library": str(candidate)})
                manifest_temp = _unique_temp(destination_dir, "manifest", ".tmp")
                manifest_temp.write_text(payload)
                os.replace(manifest_temp, manifest)
                manifest_temp = None
            except TimeoutError:
                raise
            except OSError as error:
                return False, f"publication_failed:{error}", None
            published = True
            _prune_engine_cache(destination_dir, keep_key=candidate,
                                max_artifacts=MAX_CACHED_ARTIFACTS)
            return True, "", candidate
    finally:
        if snapshot_dir is not None:
            try:
                shutil.rmtree(snapshot_dir, ignore_errors=True)
            except TimeoutError:
                raise
            except OSError:
                pass
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except TimeoutError:
                raise
            except OSError:
                pass
        if manifest_temp is not None:
            try:
                manifest_temp.unlink(missing_ok=True)
            except TimeoutError:
                raise
            except OSError:
                pass
        if candidate is not None and not published:
            _discard_path(candidate)
            _discard_path(_manifest_path(candidate))


def _memo_key(digest: bytes, compiler: str | None,
              identity: str | None) -> tuple:
    """Memoize by source, resolved compiler identity, flags and cache root.

    ``PATH`` is intentionally not part of this key: it only selects the
    compiler, and the resolved identity already captures that choice.
    """
    return (
        digest, str(cache_root()), compiler, identity,
        tuple(compiler_flags()),
    )


def build_accel(*, check_only: bool = False, force: bool = False) -> dict:
    """Compile/validate/load status for client Agents.

    ``check_only`` performs discovery and cache inspection but never invokes a
    compiler and never writes the cache.  ``force`` publishes a *new*
    generation instead of replacing a path that a process may have loaded.
    """
    try:
        data, files = source_snapshot()
    except TimeoutError:
        raise
    except OSError as error:
        return _failure("source_unavailable", detail=str(error), check_only=check_only)
    compiler, reason = discover_compiler()
    identity = None
    if compiler is not None:
        identity = _compiler_identity(compiler, persist=not check_only)
    key = _memo_key(data, compiler, identity)
    # A success memo is reusable only while its exact artifact still exists
    # and the current compiler identity is byte-identical to the memoized one.
    if not force and not check_only:
        with _MEMO_LOCK:
            cached_status = _STATUS_MEMO.get(key)
        if cached_status is not None:
            path = cached_status.get("cache_path")
            if (cached_status.get("ok") and path and compiler is not None
                    and cached_status.get("compiler_identity") == identity):
                with accel_cache.cache_lock(cache_root()):
                    if Path(path).is_file():
                        _touch_hit(Path(path))
                        _prune_engine_cache(
                            cache_root(), keep_key=Path(path),
                            max_artifacts=MAX_CACHED_ARTIFACTS)
                        result = dict(cached_status)
                        result["cache_hit"] = True
                        return result
    if compiler is None:
        status = _failure(reason, check_only=check_only)
        with _MEMO_LOCK:
            _STATUS_MEMO[key] = status
        return dict(status)
    flags = compiler_flags()
    digest = cache_digest(data, compiler, flags, identity)
    digest_hex = digest.hex() if isinstance(digest, bytes) else digest
    existing = _read_manifest(digest_hex)
    if check_only:
        return {
            "ok": True,
            "backend": "c_native" if existing is not None else "python_fallback",
            "reason": None if existing is not None else "library_not_built",
            "detail": "", "compiler": compiler,
            "compiler_identity": identity, "flags": flags,
            "digest": digest_hex,
            "cache_path": None if existing is None else str(existing),
            "cache_hit": existing is not None, "check_only": True, "abi": 2,
            "toolchain_ready": True, "library_ready": existing is not None,
        }
    if existing is not None and not force:
        # A hit must become the newest entry before retention evicts the
        # oldest; a concurrent publisher cannot delete the hit between here
        # and a caller's ``ctypes.CDLL`` because both hold this same lock.
        with accel_cache.cache_lock(cache_root()):
            existing = _read_manifest(digest_hex)
            if existing is not None:
                _touch_hit(existing)
                _prune_engine_cache(
                    cache_root(), keep_key=existing,
                    max_artifacts=MAX_CACHED_ARTIFACTS)
                status = {
                    "ok": True, "backend": "c_native", "reason": None,
                    "detail": "", "compiler": compiler,
                    "compiler_identity": identity, "flags": flags,
                    "digest": digest_hex, "cache_path": str(existing),
                    "cache_hit": True, "check_only": False, "abi": 2,
                    "toolchain_ready": True, "library_ready": True,
                }
                with _MEMO_LOCK:
                    _STATUS_MEMO[key] = status
                return dict(status)
    built, detail, candidate = _publish(compiler, flags, cache_root(), files,
                                        digest_hex)
    if not built:
        status = _failure("compile_failed", compiler=compiler, identity=identity,
                          flags=flags, digest=digest_hex, detail=detail)
        with _MEMO_LOCK:
            _STATUS_MEMO[key] = status
        return dict(status)
    status = {
        "ok": True, "backend": "c_native", "reason": None, "detail": "",
        "compiler": compiler, "compiler_identity": identity, "flags": flags,
        "digest": digest_hex, "cache_path": str(candidate), "cache_hit": False,
        "check_only": False, "abi": 2, "toolchain_ready": True,
        "library_ready": True,
    }
    with _MEMO_LOCK:
        _STATUS_MEMO[key] = status
    return dict(status)


def reset_build_memo() -> None:
    """Drop process memoization (tests that change HOME/cache only)."""
    with _MEMO_LOCK:
        _STATUS_MEMO.clear()
        _IDENTITY_MEMO.clear()
