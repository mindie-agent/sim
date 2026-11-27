"""Structure graph export package.

[Role] Expose canonical model structure graph export, physical projection,
  and native contract modules without circular top-level initialization.
[Invariants] Lazy forwarding preserves exact ``mindie_sim.structure`` API
  while allowing ``mindie_sim.structure.native`` to load independently.
[Timeouts & Locks] Pure import forwarding; no I/O or locks.
[Side-Effect Policy] Read-only module forwarding.
"""
from __future__ import annotations

import importlib
from typing import Any


def __getattr__(name: str) -> Any:
    _export = importlib.import_module("mindie_sim.structure.export")
    if hasattr(_export, name):
        value = getattr(_export, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    _export = importlib.import_module("mindie_sim.structure.export")
    return sorted(set(globals()) | set(dir(_export)))
