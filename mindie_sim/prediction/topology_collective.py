"""Topology collective physical recipe.

[Role] Lower a routed collective into pack, directed peer phases, reduction and
  unpack phases on the shared link and memory resources.
[Invariants] Per-rank wire work comes from the topology plan; this recipe never
  re-applies a bus factor. AllReduce retains two dependent phases
  (ReduceScatter then AllGather); other collectives retain one peer phase.
[Timeouts & Locks] Bounded phase count; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from dataclasses import dataclass

from .physical_recipe_base import (IO_HBM_CATEGORY, Phase, build_plan,
                                   io_weight, phased_program)
from .program import KernelProgram, Stage

ZERO_FLOAT = 0.0
ONE_FLOAT = 1.0
HALF_SHARE = 0.5
PACK_READ_SHARE = 0.25
UNPACK_WRITE_SHARE = 0.25
REDUCE_VECTOR_SHARE = 0.50
ALLREDUCE_FAMILIES = frozenset({"allreduce", "tp_allreduce"})
SINGLE_PHASE_FAMILIES = frozenset({
    "allgather", "reduce_scatter", "alltoall", "engram_broadcast", "engram_ids",
})


@dataclass(frozen=True)
class TopologyCollectiveRecipe:
    """Class E: direct-mesh or hierarchical collective phase plan."""

    traffic_rewrite_capable: bool = True

    def build(self, op, geometry):
        return build_plan(self, op, geometry)

    def lower(self, context):
        family = str(getattr(context.op, "family", ""))
        read_share = io_weight(context, "hbm")
        write_share = io_weight(context, "hbm_write")
        if family in ALLREDUCE_FAMILIES:
            phases = (
                Phase("pack", {IO_HBM_CATEGORY: read_share,
                               "mte2": ONE_FLOAT}),
                Phase("reduce_scatter", {"link": HALF_SHARE,
                                         "vector": REDUCE_VECTOR_SHARE},
                      after=("pack",)),
                Phase("all_gather", {"link": HALF_SHARE,
                                     "vector": REDUCE_VECTOR_SHARE},
                      after=("reduce_scatter",)),
                Phase("unpack_store", {
                    IO_HBM_CATEGORY: write_share,
                    "mte3": ONE_FLOAT,
                }, after=("all_gather",)),
            )
            return _with_setup(context, phases)
        phases = (
            Phase("pack", {IO_HBM_CATEGORY: read_share,
                           "mte2": ONE_FLOAT}),
            Phase("peer_exchange", {"link": ONE_FLOAT},
                  after=("pack",)),
            Phase("unpack_store", {
                IO_HBM_CATEGORY: write_share,
                "mte3": ONE_FLOAT,
            }, after=("peer_exchange",)),
        )
        return _with_setup(context, phases)


def _with_setup(context, phases):
    program = phased_program(context, phases)
    if context.setup_ms > ZERO_FLOAT:
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("body", after=("setup",), program=program),
        ), minimum_ms=context.minimum_ms)
    return program


CollectiveRecipe = TopologyCollectiveRecipe
