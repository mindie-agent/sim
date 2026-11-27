"""Built-in ``diffusion-native`` runtime stack plugin.

[Role] Export ``DiffusionNativeStackAdapter``, ``DiffusionWorkloadConfig``,
  and its manifest path.
[Invariants] Zero imports from ``export/``, ``compose/``, or ``stacks.registry``.
[Timeouts & Locks] Pure module re-export; no I/O or locks.
[Side-Effect Policy] None.
"""
from __future__ import annotations

from .adapter import (
    DiffusionNativeStackAdapter,
    DiffusionWorkloadConfig,
    MANIFEST_PATH,
)

__all__ = [
    "DiffusionNativeStackAdapter",
    "DiffusionWorkloadConfig",
    "MANIFEST_PATH",
]
