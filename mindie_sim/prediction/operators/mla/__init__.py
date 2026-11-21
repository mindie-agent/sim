"""MLA semantic, dense-latent, and decode operator implementations.

[Role] Governed operator package owning ``mla`` device kernel recipes.
[Invariants] Single-owner operator package; no hardware capability sniffing.
[Timeouts & Locks] Pure namespace package; no I/O or locks.
[Side-Effect Policy] Zero import side effects.
"""
