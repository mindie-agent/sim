"""Private exact-period detection for the existing finite pipeline Engine.

[Role] Detect a genuinely repeated *settled* Engine structure and return a
  precise, bounded skip description. The consumer owns Engine mutations.
  The package imports no Engine, scheduling or program code.
[Invariants] A cycle requires (a) exact structural identity of the supplied
  normalized state, (b) an identical active cohort/phase class, (c) two
  consecutive equal measured periods (each aggregated from real intervals),
  and (d) the same per-run issue, retire and sequence strides. Phase remaining
  bounds the number of cycles and stays outside the structural key. Floating
  rounding is tolerated only at a few ulps for measured time/service values;
  structural state values are compared deeply and exactly.  A cheap structural
  fingerprint may only fast-reject; a match always uses exact identity or
  full equality.  Mutable/opaque external state keeps the full validation path.
[Timeouts & Locks] Pure CPU, bounded observation history, no threads, sleeps,
  I/O, retries or engine access.  Unsafe or inconsistent input is rejected
  with ``PeriodicContractError`` rather than silently becoming a skip.
[Side-Effect Policy] ``CycleDetector.observe`` mutates only this detector's
  private observation history.  Returned records are frozen.  The caller must
  explicitly apply a ``CycleSkip`` and remains responsible for real event,
  allocation, peak, service, busy and sequence accounting.
"""
from __future__ import annotations

from .detector import CycleDetector, _CycleEvidence, _HistoryFrame, _Observation
from .engine_state import split_engine_state
from .frontier import (
    Cursor,
    CycleCorrespondence,
    CycleSkip,
    IntervalDelta,
    RunCorrespondence,
    SettledFrontier,
    WindowTile,
)
from .interner import (
    CanonicalState,
    StateInterner,
    TemplateRef,
    metrics,
    reset_metrics,
    reset_state_caches,
)
from .snapshots import PeriodicContractError, structural_difference, structural_equal

__all__ = [
    "CanonicalState",
    "Cursor",
    "CycleCorrespondence",
    "CycleDetector",
    "CycleSkip",
    "IntervalDelta",
    "PeriodicContractError",
    "RunCorrespondence",
    "SettledFrontier",
    "StateInterner",
    "TemplateRef",
    "WindowTile",
    "metrics",
    "reset_metrics",
    "reset_state_caches",
    "split_engine_state",
    "structural_difference",
    "structural_equal",
]
