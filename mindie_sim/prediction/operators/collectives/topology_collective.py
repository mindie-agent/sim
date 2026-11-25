"""Topology collective physical recipe.

[Role] Lower a routed collective into pack, directed peer phases, reduction and
  unpack phases on the shared link and memory resources.
[Invariants] Per-rank wire work comes from the topology plan; this recipe never
  re-applies a bus factor. A topology-declared direct mesh is consumed as its
  physical plan: ReduceScatter then AllGather for AllReduce, one peer phase for
  ReduceScatter/AllGather, each issuing P/W per directed peer concurrently.
  Non-mesh collectives retain the ring/NHR/hardware-scheduled phase split. AllReduce retains
  two dependent phases; other collectives retain one peer phase.
[Timeouts & Locks] Bounded phase count; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from mindie_sim.prediction.physical_recipe_base import (IO_HBM_CATEGORY, Phase, build_plan,
                                   io_weight, phased_program)
from mindie_sim.prediction.program import KernelProgram, Stage

HALF_SHARE = 0.5
REDUCE_VECTOR_SHARE = 0.50
MESH_ALGORITHM = "direct_mesh"
ALLREDUCE_FAMILIES = frozenset({"allreduce", "tp_allreduce"})


def _pack_phase(read_share):
    return Phase("pack", {IO_HBM_CATEGORY: read_share, "mte2": 1.0})


def _unpack_phase(write_share, after):
    return Phase("unpack_store", {
        IO_HBM_CATEGORY: write_share,
        "mte3": 1.0,
    }, after=after)


def _mesh_phases(context, phases, family):
    """Physical phase weights around the plan-owned peer transports."""
    read_share = io_weight(context, "hbm_read")
    write_share = io_weight(context, "hbm_write")
    result = [_pack_phase(read_share)]
    reduce = (REDUCE_VECTOR_SHARE if family in ALLREDUCE_FAMILIES
              else 1.0)
    for name in phases:
        result.append(Phase(name, {"vector": reduce}, after=(result[(-1)].name,)))
    result.append(_unpack_phase(write_share, after=(result[(-1)].name,)))
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
        from mindie_sim.prediction.collectives.band_plan import BandCollectivePlan

        binding = context.collective_plan
        if getattr(binding, "algorithm", None) == MESH_ALGORITHM:
            return _lower_mesh(context, binding())
        family = str(getattr(context.op, "family", ""))
        read_share = io_weight(context, "hbm_read")
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
            base = _with_setup(context, phases)
        else:
            phases = (
                _pack_phase(read_share),
                Phase("peer_exchange", {"link": 1.0},
                      after=("pack",)),
                _unpack_phase(write_share, after=("peer_exchange",)),
            )
            base = _with_setup(context, phases)
        bound_plan = getattr(binding, "plan", None)
        if isinstance(bound_plan, BandCollectivePlan):
            return bound_plan.lower_with_local_work(context, base)
        return base


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
    # The setup stage is part of the invocation, so the outer program owns the
    # inclusive floor. Keep the nested body floor-free: charging it again after
    # setup would add setup_ms to the declared completion response.
    program = phased_program(
        context, phases, minimum_ms=0.0 if context.setup_ms > 0.0 else None)
    if context.setup_ms > 0.0:
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("body", after=("setup",), program=program),
        ), minimum_ms=context.minimum_ms)
    return program


CollectiveRecipe = TopologyCollectiveRecipe
