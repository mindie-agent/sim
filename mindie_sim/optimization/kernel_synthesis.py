"""TOMBSTONE — deleted in Step 5 once W2 (export/bundle.py) & W5 (sweep/pd.py) switch.

[Role] Temporary import-only tombstone for the retired optimization.kernel_synthesis module.
[Invariants] Calling any exported symbol raises NotImplementedError loudly (zero legacy execution).
[Timeouts & Locks] None.
[Side-Effect Policy] Pure fail-loud stubs; zero runtime side effects.
"""
from __future__ import annotations

_MSG = (
    "LEGACY_PURGED: optimization.kernel_synthesis deleted; wire canonical W3a producers "
    "(optimization.ladder.solve_target_step_receipt, optimization.codesign.evaluate_macro_wall_trigger, "
    "rewrite.schema.load_rewrite_plan, templates.contracts)."
)


class TargetOpNode:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_MSG)


def solve_target_graph(*args, **kwargs):
    raise NotImplementedError(_MSG)


def evaluate_macro_wall_trigger(*args, **kwargs):
    raise NotImplementedError(_MSG)


def load_rewrite_plan(*args, **kwargs):
    raise NotImplementedError(_MSG)


__all__ = (
    "TargetOpNode",
    "evaluate_macro_wall_trigger",
    "load_rewrite_plan",
    "solve_target_graph",
)
