"""Topology collective physical recipe.

[Role] Lower a routed collective into pack, directed peer phases, reduction and
  unpack phases on the shared link and memory resources.
[Invariants] Per-rank wire work comes from the topology plan; this recipe never
  re-applies a bus factor. A topology-declared direct mesh is consumed as its
  physical plan: ReduceScatter then AllGather for AllReduce, one peer phase for
  ReduceScatter/AllGather, each issuing P/W per directed peer concurrently.
  Non-mesh collectives retain the ring/NHR/CCU phase split. AllReduce retains
  two dependent phases; other collectives retain one peer phase.
[Timeouts & Locks] Bounded phase count; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from .physical_recipe_base import (IO_HBM_CATEGORY, Phase, build_plan,
                                   io_weight, phased_program)
from .program import KernelProgram, Stage

ZERO_FLOAT = 0.0
ONE_FLOAT = 1.0
HALF_SHARE = 0.5
PACK_READ_SHARE = 0.25
UNPACK_WRITE_SHARE = 0.25
REDUCE_VECTOR_SHARE = 0.50
MESH_ALGORITHM = "direct_mesh"
LAST = -1
ALLREDUCE_FAMILIES = frozenset({"allreduce", "tp_allreduce"})
SINGLE_PHASE_FAMILIES = frozenset({
    "allgather", "reduce_scatter", "alltoall", "engram_broadcast", "engram_ids",
})


def _pack_phase(read_share):
    return Phase("pack", {IO_HBM_CATEGORY: read_share, "mte2": ONE_FLOAT})


def _unpack_phase(write_share, after):
    return Phase("unpack_store", {
        IO_HBM_CATEGORY: write_share,
        "mte3": ONE_FLOAT,
    }, after=after)


def _mesh_phases(context, phases, family):
    """Physical phase weights around the plan-owned peer transports."""
    read_share = io_weight(context, "hbm")
    write_share = io_weight(context, "hbm_write")
    result = [_pack_phase(read_share)]
    reduce = (REDUCE_VECTOR_SHARE if family in ALLREDUCE_FAMILIES
              else ONE_FLOAT)
    for name in phases:
        result.append(Phase(name, {"vector": reduce}, after=(result[LAST].name,)))
    result.append(_unpack_phase(write_share, after=(result[LAST].name,)))
    return tuple(result)


@dataclass(frozen=True)
class TopologyCollectiveRecipe:
    """Class E: direct-mesh or hierarchical collective phase plan."""

    traffic_rewrite_capable: bool = True
    #: The topology may declare a direct full-mesh plan for this invocation.
    mesh_capable: bool = True

    def build(self, op, geometry):
        return build_plan(self, op, geometry)

    def lower(self, context):
        binding = context.collective_plan
        if getattr(binding, "algorithm", None) == MESH_ALGORITHM:
            return _lower_mesh(context, binding())
        family = str(getattr(context.op, "family", ""))
        read_share = io_weight(context, "hbm")
        write_share = io_weight(context, "hbm_write")
        if family in ALLREDUCE_FAMILIES:
            phases = (
                _pack_phase(read_share),
                Phase("reduce_scatter", {"link": HALF_SHARE,
                                         "vector": REDUCE_VECTOR_SHARE},
                      after=("pack",)),
                Phase("all_gather", {"link": HALF_SHARE,
                                     "vector": REDUCE_VECTOR_SHARE},
                      after=("reduce_scatter",)),
                _unpack_phase(write_share, after=("all_gather",)),
            )
            return _with_setup(context, phases)
        phases = (
            _pack_phase(read_share),
            Phase("peer_exchange", {"link": ONE_FLOAT},
                  after=("pack",)),
            _unpack_phase(write_share, after=("peer_exchange",)),
        )
        return _with_setup(context, phases)


def _lower_mesh(context, plan):
    """Submit the selected plan's dependent phases; never re-split ring demand."""
    plan.validate_path_model("direct")
    family = str(getattr(context.op, "family", ""))
    # The plan owns all link work.  Keep only local packaging/reduction work in
    # the stages that wrap the physical peer programs, so no byte is counted
    # twice.
    link_names = frozenset(plan.wire_by_resource())
    work = {name: value for name, value in dict(context.work).items()
            if name not in link_names}
    phases = _mesh_phases(context, tuple(plan.phase_names), family)
    base = phased_program(replace(context, work=work), phases)
    stages = []
    for stage in base.stages:
        if stage.name in plan.phase_names:
            # A dependent phase is a real program boundary: its startup and
            # every directed peer message come from the topology plan.
            stage = replace(stage, program=plan.phase_program(stage.name),
                            keep_empty=True)
        stages.append(stage)
    return KernelProgram(tuple(stages), minimum_ms=context.minimum_ms)


def _with_setup(context, phases):
    program = phased_program(context, phases)
    if context.setup_ms > ZERO_FLOAT:
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("body", after=("setup",), program=program),
        ), minimum_ms=context.minimum_ms)
    return program


CollectiveRecipe = TopologyCollectiveRecipe
