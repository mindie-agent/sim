"""[Role] Translate native model configurations into structural records.
[Invariants] Importing this package performs no simulation or device work.
[Timeouts & Locks] No waits or locks.
[Side-Effect Policy] No I/O.
"""
from __future__ import annotations

from typing import Any


def load_model(*args: Any, **kwargs: Any) -> Any:
    """Delegate catalog loading to ``mindie_sim.model_catalog.load_model``."""
    from mindie_sim.model_catalog import load_model as _load_model

    return _load_model(*args, **kwargs)


def normalize_model(*args: Any, **kwargs: Any) -> Any:
    """Delegate structural normalization to ``mindie_sim.model_catalog.normalize_model``."""
    from mindie_sim.model_catalog import normalize_model as _normalize_model

    return _normalize_model(*args, **kwargs)


__all__ = ["load_model", "normalize_model"]
