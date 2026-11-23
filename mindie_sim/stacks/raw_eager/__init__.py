"""Built-in ``raw-eager`` baseline runtime stack plugin.

[Role] Export ``RawEagerStackAdapter`` and its manifest path.
[Invariants] Zero imports from ``export/``, ``compose/``, or ``stacks.registry``.
[Timeouts & Locks] Pure module re-export; no I/O or locks.
[Side-Effect Policy] None.
"""
from __future__ import annotations

from .adapter import MANIFEST_PATH, RawEagerStackAdapter

__all__ = ["MANIFEST_PATH", "RawEagerStackAdapter"]
