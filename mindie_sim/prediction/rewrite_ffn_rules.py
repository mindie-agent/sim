"""Temporary Wave-7 compat shim forwarding to ``mindie_sim.prediction.sharding.ffn_rules``.

[Role] Preserve untouched ``operators/moe_routing/moe_gating_fusion.py`` imports until Wave-8 merge.
[Invariants] Zero standalone logic; pure re-export.
[Timeouts & Locks] None.
[Side-Effect Policy] Read-only.
"""
from mindie_sim.prediction.sharding.ffn_rules import *  # noqa: F401,F403
