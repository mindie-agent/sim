"""Strided multi-stream low-rank hyper-connection vector recipe.

[Role] Provide the capability-scoped AIV vector plan for the five low-rank
  hyper-connection vector recipes (``hc_inverse_rms``,
  ``hc_normalize_scale_bias_gate``, ``hc_sinkhorn``, ``hc_pre_mix``,
  ``hc_post_mix``).
[Invariants] Work equals ``work_from_op(op)`` across all active AIV owners.
  Capability scoping checks declared device UB capacity and ALU lanes via
  geometry/datapath rather than hardware SKU names or ``service_modes``.
[Timeouts & Locks] O(active AIV owners) pure plan construction; no I/O or locks.
[Side-Effect Policy] Pure immutable ``OperatorPlan`` construction.
"""
from __future__ import annotations

from mindie_sim.schema.operators import OpSpec
from .mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from .program import KernelProgram, ParallelProgram, Stage

LOWRANK_HC_RECIPES: tuple[str, ...] = (
    "hc_gate_mix",
    "hc_combine",
    "grouped_gemma_rmsnorm",
    "hc_combine_rmsnorm",
)
REGBASE_UB_MIN_BYTES = 248 * 1024
LOWRANK_HC_STRIDED_HBM_EFF = 0.14
LOWRANK_HC_SETUP_MS = 0.053


def _is_regbase_capable(geometry: Geometry) -> bool:
    ub_cap = int((geometry.buffers or {}).get("ub", 0))
    units = geometry.resource_units or {}
    return ub_cap >= REGBASE_UB_MIN_BYTES and "reg_alu_lanes" in units


def build_lowrank_hc_vector(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Build a strided multi-stream HC vector plan conserving ``work_from_op``."""
    from .physical_recipe_base import build_plan, work_from_op
    from .vector_pipeline import AIVTiledVectorAndGatherRecipe

    if not _is_regbase_capable(geometry):
        return build_plan(AIVTiledVectorAndGatherRecipe(), op, geometry)
    base = work_from_op(op)
    rows = max(1, int(op.local_shape[0]) if op.local_shape else 1)
    read = float(op.mte2_bytes)
    write = float(op.mte3_bytes)
    hbm = float(op.hbm_bytes)
    vector = float(op.vector_ops)
    aiv_units = max(1, int((geometry.resource_units or {}).get("vector", 1)))
    active_aiv = min(aiv_units, rows)
    share = 1.0 / float(active_aiv)
    regbase_setup = LOWRANK_HC_SETUP_MS

    def program(context):
        caps: dict[str, float] = {}
        if "hbm" in context.resource_names:
            hbm_phys = context.resource_names["hbm"]
            hbm_rate = float(context.rates.get(hbm_phys, 0.0))
            if hbm_rate > 0.0:
                caps[hbm_phys] = (LOWRANK_HC_STRIDED_HBM_EFF * hbm_rate) * share
        has_dram_aic = context.service_rate("mte2_aic_dram") is not None
        setup = regbase_setup if has_dram_aic else 0.0

        def per_aiv(aiv_idx: int):
            if aiv_idx >= active_aiv:
                return KernelProgram(())
            owned = context.owned(aiv_idx, aiv_units, ("mte2_aiv", "vector", "mte3"))
            res: dict[str, float] = {}
            if hbm:
                res["hbm"] = hbm * share
            if read:
                res["mte2"] = read * share
            if vector:
                res["vector"] = vector * share
            if write:
                res["mte3"] = write * share
            s_setup = Stage("hc_strided_setup", latency=setup, keep_empty=setup <= 0.0)
            s_body = Stage(
                "hc_strided_body",
                context.bind(Work(resources=res)),
                after=("hc_strided_setup",),
                placement=owned,
                caps=caps,
            )
            return KernelProgram((s_setup, s_body))

        return ParallelProgram(tuple(per_aiv(i) for i in range(aiv_units)))

    return OperatorPlan(base, program, {
        "execution_class": "vector_aiv",
        "hc_variant": str(op.recipe),
    })


class ScopedLowRankHcVectorRecipe(PlannedRecipe):
    """Scope strided multi-stream HC vector planning strictly to RegBase devices."""

    def prepare(self, op, geometry):
        if not _is_regbase_capable(geometry):
            from .physical_recipe_registry import VECTOR
            return op, VECTOR
        return super().prepare(op, geometry)


def recipes() -> dict[str, PlannedRecipe]:
    """Return physical recipe bindings for the five low-rank HC vector ops."""
    plan = ScopedLowRankHcVectorRecipe(
        build_lowrank_hc_vector, traffic_rewrite_capable=True
    )
    return {name: plan for name in LOWRANK_HC_RECIPES}


__all__ = [
    "LOWRANK_HC_RECIPES",
    "ScopedLowRankHcVectorRecipe",
    "build_lowrank_hc_vector",
    "recipes",
]
