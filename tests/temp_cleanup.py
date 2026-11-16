"""Process- and session-owned temporary directories for the maintained tests.

[Role] Create test roots and guarantee their removal at pytest-session or
  process teardown even when an individual test misses its own ``finally``.
[Invariants] Every path comes from ``tempfile.mkdtemp``; cleanup ignores
  already-missing paths, never deletes an unregistered root, and drops a path
  from the ownership list only after it is verifiably gone.
[Timeouts & Locks] Pure in-process ownership list; no locks or background work.
[Side-Effect Policy] Deletes only directories created through ``tempdir``.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

_ROOTS: list[Path] = []


def tempdir(prefix: str, *, directory: str | Path | None = None) -> Path:
    """Create a tracked temporary directory honoring the isolated TMPDIR."""
    path = Path(tempfile.mkdtemp(
        prefix=prefix, dir=None if directory is None else str(directory)))
    _ROOTS.append(path)
    return path


def cleanup() -> None:
    """Remove every still-registered root; safe to call repeatedly.

    A path is dropped from the ownership list only after it no longer exists,
    so a transient Windows handle/share violation is retried by the next
    ``pytest_sessionfinish`` or ``atexit`` call instead of leaking forever.
    """
    pending: list[Path] = []
    for path in _ROOTS:
        if not os.path.lexists(path):
            continue
        shutil.rmtree(path, ignore_errors=True)
        if os.path.lexists(path):
            pending.append(path)
    _ROOTS[:] = pending


atexit.register(cleanup)
