"""MoE DeepEP and inter-die token transport caller operators.

[Role] Governed operator package owning ``moe_transport`` device kernel recipes.
[Invariants] Single-owner operator package; no hardware capability sniffing.
[Timeouts & Locks] Pure namespace package; no I/O or locks.
[Side-Effect Policy] Zero import side effects.
"""
