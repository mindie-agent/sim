"""Temporary Wave-7 compat shim forwarding to ``mindie_sim.prediction.sharding.context_parallel``.

[Role] Preserve untouched ``operators/<family>/`` imports until Wave-8 merge.
[Invariants] Zero standalone logic; pure re-export.
[Timeouts & Locks] None.
[Side-Effect Policy] Read-only.
"""
from mindie_sim.prediction.sharding.context_parallel import *  # noqa: F401,F403
from mindie_sim.prediction.sharding import context_parallel as graph  # noqa: F401
