"""Locked ``ctypes`` loader for the optional C11 engine kernel.

[Role] Turn a cache status into one validated in-process library handle while
  holding the engine cache's cross-process advisory lock.
[Invariants] A path returned by ``build_accel`` is mapped before any concurrent
  publisher can prune it.  A stale path that vanished after a standalone build,
  or a file that no longer loads, is rebuilt once with ``force=True`` and
  mapped again.  ABI version, probe status and golden cells are checked before
  the handle is memoized.
[Timeouts & Locks] The cache lock is held across ``build_accel`` and
  ``ctypes.CDLL``; the loader nests inside an already-locked attach.  Lock
  timeouts raise ``TimeoutError`` and are never swallowed.
[Side-Effect Policy] Reads and maps one immutable cache library; a stale hit
  may publish a fresh generation under the cache directory.  No model,
  hardware or network access.
"""
from __future__ import annotations

import ctypes
import os

from . import accel_build, accel_cache
from .accel_tables import ABI_VERSION, PROBE_GOLDEN, install_signatures

_LOADED: dict[str, ctypes.CDLL] = {}
ABI_PROBE_CELLS = 16
ABI_PROBE_RC = 0
MAX_LOADED_LIBRARIES = 8


class AccelUnavailable(RuntimeError):
    """Explicit native backend requested without a usable library."""
    def __init__(self, reason: str, status: dict | None = None) -> None:
        super().__init__(f"accel_backend_unavailable:{reason}")
        self.reason = reason
        self.status = dict(status or {})


def _rebuild_for_load(status: dict, *, force: bool = True) -> dict:
    """Rebuild once under the cache lock when a hit path went stale."""
    try:
        rebuilt = accel_build.build_accel(force=force)
    except TimeoutError:
        raise
    except (OSError, RuntimeError) as error:
        raise AccelUnavailable(f"build_unavailable:{error}", status) from error
    if not rebuilt.get("ok"):
        raise AccelUnavailable(
            rebuilt.get("reason") or "native_build_unavailable", rebuilt)
    return rebuilt


def _check_library(library: ctypes.CDLL, status: dict) -> None:
    try:
        install_signatures(library)
        if int(library.msim_abi_version()) != ABI_VERSION:
            raise AccelUnavailable("abi_version_mismatch", status)
        probe = (ctypes.c_uint64 * ABI_PROBE_CELLS)()
        if int(library.msim_abi_probe(probe, ABI_PROBE_CELLS)) != ABI_PROBE_RC:
            raise AccelUnavailable("abi_probe_failed", status)
        if tuple(int(x) for x in probe) != PROBE_GOLDEN:
            raise AccelUnavailable("abi_probe_mismatch", status)
    except TimeoutError:
        raise
    except OSError as error:
        raise AccelUnavailable(f"library_load_failed:{error}", status) from error
    except AttributeError as error:
        raise AccelUnavailable(f"abi_symbol_missing:{error}", status) from error


def load_library(status: dict) -> ctypes.CDLL:
    """Map one cached library while holding the cross-process cache lock.

    The lock stays held through ``ctypes.CDLL`` so a concurrent publish/prune
    in another thread or process cannot unlink the path before the inode is
    mapped.  A stale path (vanished or no longer loadable) is rebuilt once with
    ``force=True`` under the same lock and mapped again.
    """
    with accel_cache.cache_lock(accel_build.cache_root()):
        current = status
        path = current.get("cache_path")
        if path:
            loaded = _LOADED.get(path)
            if loaded is not None:
                return loaded
        rebuilt = retried = False
        while True:
            if not path or not os.path.isfile(path):
                if rebuilt:
                    raise AccelUnavailable("library_missing", current)
                rebuilt = True
                current = _rebuild_for_load(current)
                path = current.get("cache_path")
                continue
            try:
                library = ctypes.CDLL(path)
            except OSError as error:
                if retried:
                    raise AccelUnavailable(
                        f"library_load_failed:{error}", current) from error
                retried = rebuilt = True
                current = _rebuild_for_load(current)
                path = current.get("cache_path")
                continue
            _check_library(library, current)
            if len(_LOADED) >= MAX_LOADED_LIBRARIES:
                _LOADED.pop(next(iter(_LOADED)))
            _LOADED[path] = library
            return library


def load_accel_cdll(status: dict | None = None) -> tuple[dict, ctypes.CDLL]:
    """Build (unless supplied) and map under one cache lock.

    This is the loader entry point for callers that do not already own a
    status; it removes the standalone ``build_accel`` -> ``CDLL`` window.
    """
    with accel_cache.cache_lock(accel_build.cache_root()):
        current = status
        if current is None:
            try:
                current = accel_build.build_accel()
            except TimeoutError:
                raise
            except (OSError, RuntimeError) as error:
                raise AccelUnavailable(f"build_unavailable:{error}") from error
        if not current.get("ok"):
            raise AccelUnavailable(
                current.get("reason") or "native_build_unavailable", current)
        return current, load_library(current)
