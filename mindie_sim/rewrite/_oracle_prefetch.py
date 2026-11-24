"""Prefetch binding and compatible multi-stream event wiring for rewrite replay.

[Role] Materialize declared ``WeightPrefetch`` leases into causal prefetch and
  release operators while aligning synthetic ops with active ``StreamPlan``
  assignments and explicit cross-stream ``StreamEvent`` edges.
[Invariants] Every prefetch op depends causally on its window anchor and runs on
  its target operator's stream; cross-stream anchors receive explicit wait
  events so prefetch and multi-stream schedules remain sound and race-free.
[Timeouts & Locks] Pure bounded graph rewriting; no I/O, threads or locks.
[Side-Effect Policy] Returns new operator tuples, recipe books and stream plans.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.prediction.fusion.bridge import (
    CanonicalPlan, add_prefetch, apply_prefetch_rewrite, settle_prefetch_rewrite,
)
from mindie_sim.prediction.fusion.pool import register_recipe
from mindie_sim.prediction.streams import MAIN_STREAM, StreamEvent
from mindie_sim.prediction.execution.weight_prefetch import PrefetchDecision
from mindie_sim.schema.common import ModelingError

PREFETCH_PRIORITY = 10
DISABLED_PRIMITIVE_TABLE_TOKEN = "__primitive_table_disabled__"


def bind_prefetch_ops(case, replica, ops, book, leases, hardware):
    """Lower prefetch leases onto ``ops`` and register replica-scoped recipes."""
    del case, replica, ops, book, leases, hardware
    raise NotImplementedError(
        "LEGACY_PURGED: scope prefetch lease bindings and lookups by (replica, op_id)"
    )


def bind_prefetch_streams(graphs, streams, leases):
    """Attach generated prefetch/release ops and cross-stream anchor events to ``streams``."""
    if not streams or not leases:
        return tuple(streams)
    del graphs
    raise NotImplementedError(
        "LEGACY_PURGED: scope prefetch stream bindings and lookups by (replica, op_id)"
    )


def die_placements_for_case(case) -> tuple[tuple[int, tuple[int, int]], ...]:
    """Return canonical ``(die_rank, (replica, pp_stage))`` tuples for a case."""
    parallel = case.runtime.parallel
    width = int(parallel.tp) * int(parallel.pcp)
    pp = int(parallel.pp)
    limit = int(parallel.dp) * pp * width
    return tuple(
        (int(die.rank), (idx // (pp * width), (idx // width) % pp))
        for idx, die in enumerate(case.machine.dies)
        if idx < limit
    )


def primitive_table_digest_token(primitive_table):
    """Return a deterministic content digest token for ``primitive_table``."""
    if primitive_table is False:
        return DISABLED_PRIMITIVE_TABLE_TOKEN
    if primitive_table is None:
        return None
    canonical = getattr(primitive_table, "canonical_envelope_sha256", None)
    if callable(canonical):
        return canonical()
    return getattr(primitive_table, "envelope_sha256", None) or getattr(primitive_table, "entries", None)
