"""Process- and session-owned temporary directories for the maintained tests.

[Role] Create test roots and guarantee their removal at pytest-session or
  process teardown even when an individual test misses its own ``finally``.
[Invariants] Every path comes from ``tempfile.mkdtemp`` and is removed at most
  once; cleanup ignores already-missing paths and never deletes an unregistered
  root.
[Timeouts & Locks] Pure in-process ownership list; no locks or background work.
[Side-Effect Policy] Deletes only directories created through ``tempdir``.
"""
from __future__ import annotations

import atexit
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
    """Remove every still-registered root; safe to call repeatedly."""
    while _ROOTS:
        shutil.rmtree(_ROOTS.pop(), ignore_errors=True)


atexit.register(cleanup)
