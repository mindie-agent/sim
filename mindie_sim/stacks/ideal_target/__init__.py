"""Built-in ``ideal-target`` hardware-limit runtime stack plugin.

[Role] Export ``IdealTargetStackAdapter`` and its manifest path.
[Invariants] Zero imports from ``export/``, ``compose/``, or ``stacks.registry``.
[Timeouts & Locks] Pure module re-export; no I/O or locks.
[Side-Effect Policy] None.
"""
from __future__ import annotations

from .adapter import IdealTargetStackAdapter, MANIFEST_PATH

__all__ = ["IdealTargetStackAdapter", "MANIFEST_PATH"]
