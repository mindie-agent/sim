"""Strided multi-stream low-rank hyper-connection vector recipe.

[Role] Provide the capability-scoped AIV vector plan for the low-rank
  hyper-connection vector recipes (``hc_gate_mix``, ``hc_combine``,
  ``grouped_gemma_rmsnorm``, ``hc_combine_rmsnorm``).
[Invariants] Work equals ``work_from_op(op)`` across all active AIV owners.
  Capability scoping dispatches strictly on ``compute_arch.arch_id`` (Arch35 /
  ``npu_arch_3510``) and derives launch/HBM service strictly from
  ``PrimitiveTable`` (zero hardcoded latency floors or efficiency multipliers).
[Timeouts & Locks] O(active AIV owners) pure plan construction; no I/O or locks.
[Side-Effect Policy] Pure immutable ``OperatorPlan`` construction.
"""
from __future__ import annotations

from mindie_sim.primitives.loader import resolve_dispatch_envelope
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, Stage

LOWRANK_HC_RECIPES: tuple[str, ...] = (
    "hc_gate_mix",
    "hc_combine",
    "grouped_gemma_rmsnorm",
    "hc_combine_rmsnorm",
)
_ARCH35_IDS = frozenset({"npu_arch_3510", "dav_3510"})
FP32_WORKING_BYTES = 4
_MS_PER_SECOND = 1000.0


def _is_regbase_capable(geometry: Geometry) -> bool:
    """True strictly for declared Arch35 (``npu_arch_3510``) datapaths."""
    dp_arch = getattr(getattr(geometry, "datapath", None), "compute_arch", None)
    ca = dp_arch or getattr(geometry, "compute_arch", None)
    arch_id = str(getattr(ca, "arch_id", "") or getattr(ca, "preset", "") or "").strip().lower()
    return arch_id in _ARCH35_IDS


def _op_rows(op: OpSpec) -> int:
    if not op.local_shape:
        raise ModelingError(f"lowrank_hc_local_shape_empty:{op.op_id}")
    rows = int(op.local_shape[0])
    if rows < 0:
        raise ModelingError(f"lowrank_hc_negative_rows:{op.op_id}:{rows}")
    return rows


def _build_fused_ub_resident_hc(op: OpSpec, geometry: Geometry, base: Work) -> OperatorPlan:
    """UB-resident fused low-rank HC plan conserving ``work_from_op``."""
    rows = _op_rows(op)
    if rows == 0:
        return OperatorPlan(Work(), lambda _ctx: KernelProgram(()), {
            "execution_class": "vector_aiv",
            "hc_variant": str(op.recipe),
            "fused_ub_resident": True,
            "ub_compute_dtype": "fp32",
            "ub_working_bytes_per_elem": FP32_WORKING_BYTES,
        })
    read = float(op.mte2_bytes)
    write = float(op.mte3_bytes)
    hbm = float(op.hbm_bytes)
    vector = float(op.vector_ops)
    aiv_units = max(1, int((geometry.resource_units or {}).get("vector", 1)))
    active_aiv = min(aiv_units, rows)
    share = 1.0 / float(active_aiv)
    matrix_work = dict(base.matrix)

    def program(context):
        branches = []
        if matrix_work:
            aic_units = max(1, int(getattr(context, "effective_aic_cores", 1) or 1))
            aic_owned = context.owned(0, 1, ("cube",))
            aic_owned = {k: v[:aic_units] for k, v in aic_owned.items()}
            branches.append(KernelProgram((Stage(
                "hc_ub_cube",
                context.bind(Work(matrix=matrix_work)),
                placement=aic_owned,
            ),)))
        for aiv_idx in range(aiv_units):
            if aiv_idx >= active_aiv:
                branches.append(KernelProgram(()))
                continue
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
            branches.append(KernelProgram((Stage(
                "hc_ub_body",
                context.bind(Work(resources=res)),
                placement=owned,
            ),)))
        return ParallelProgram(tuple(branches))

    return OperatorPlan(base, program, {
        "execution_class": "vector_aiv",
        "hc_variant": str(op.recipe),
        "fused_ub_resident": True,
        "ub_compute_dtype": "fp32",
        "ub_working_bytes_per_elem": FP32_WORKING_BYTES,
    })


def build_lowrank_hc_vector(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Build an Arch35 strided multi-stream HC vector plan conserving ``work_from_op``."""
    from mindie_sim.prediction.physical_recipe_base import build_plan, work_from_op
    from mindie_sim.prediction.operators.vector.vector_pipeline import AIVTiledVectorAndGatherRecipe

    base = work_from_op(op)
    if bool(dict(op.metadata).get("fused_ub_resident", False)):
        return _build_fused_ub_resident_hc(op, geometry, base)
    if not _is_regbase_capable(geometry):
        return build_plan(AIVTiledVectorAndGatherRecipe(), op, geometry)
    rows = _op_rows(op)
    if rows == 0:
        return OperatorPlan(Work(), lambda _ctx: KernelProgram(()), {
            "execution_class": "vector_aiv",
            "hc_variant": str(op.recipe),
            "ub_compute_dtype": "fp32",
            "ub_working_bytes_per_elem": FP32_WORKING_BYTES,
        })
    read = float(op.mte2_bytes)
    write = float(op.mte3_bytes)
    hbm = float(op.hbm_bytes)
    vector = float(op.vector_ops)
    aiv_units = max(1, int((geometry.resource_units or {}).get("vector", 1)))
    active_aiv = min(aiv_units, rows)
    share = 1.0 / float(active_aiv)

    def program(context):
        caps: dict[str, float] = {}
        tbl = getattr(context, "primitive_table", None)
        dp = getattr(context, "datapath", None) or getattr(geometry, "datapath", None)
        if "hbm" in context.resource_names:
            hbm_phys = context.resource_names["hbm"]
            strided_bps = (
                context.service_rate("hbm_rw21_dram")
                or context.service_rate("hbm_rw11_dram")
            )
            if strided_bps is None and tbl is not None:
                band = tbl.cell("P01_datacopy").band("hbm_rw21_bytes_per_s")
                if band.p50 is not None and float(band.p50) > 0.0:
                    strided_bps = float(band.p50)
            if strided_bps is not None and float(strided_bps) > 0.0:
                caps[hbm_phys] = (float(strided_bps) / _MS_PER_SECOND) * share
        setup_ms = 0.0
        if tbl is not None and dp is not None:
            env = resolve_dispatch_envelope(tbl, dp)
            if env is not None:
                setup_ms = float(env.aiv_floor_ms(active_aiv))

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
            s_body = Stage(
                "hc_strided_body",
                context.bind(Work(resources=res)),
                after=("hc_strided_setup",) if setup_ms > 0.0 else (),
                placement=owned,
                caps=caps,
            )
            if setup_ms > 0.0:
                return KernelProgram((Stage("hc_strided_setup", latency=setup_ms), s_body))
            return KernelProgram((s_body,))

        return ParallelProgram(tuple(per_aiv(i) for i in range(aiv_units)))

    return OperatorPlan(base, program, {
        "execution_class": "vector_aiv",
        "hc_variant": str(op.recipe),
        "ub_compute_dtype": "fp32",
        "ub_working_bytes_per_elem": FP32_WORKING_BYTES,
    })


class ScopedLowRankHcVectorRecipe(PlannedRecipe):
    """Scope strided multi-stream HC vector planning strictly to Arch35 devices."""

    def prepare(self, op, geometry):
        if (not bool(dict(op.metadata).get("fused_ub_resident", False))
                and not _is_regbase_capable(geometry)):
            from mindie_sim.prediction.physical_recipe_registry import VECTOR
            return op, VECTOR
        return super().prepare(op, geometry)


def recipes() -> dict[str, PlannedRecipe]:
    """Return physical recipe bindings for the low-rank HC vector ops."""
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
