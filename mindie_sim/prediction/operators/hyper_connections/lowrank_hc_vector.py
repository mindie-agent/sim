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

from typing import Literal

from mindie_sim.primitives.loader import resolve_dispatch_envelope
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, Stage

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCE_REFS: tuple[SourceRef, ...] = (
    SourceRef(
        "vllm-project/vllm-ascend",
        "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        "vllm_ascend/ops/triton/qwen4_exp/hc.py",
        "28c6c8c6f26ff6c92023334eaed86d3c8606200dd27a67446cef1d624a0c1f91",
    ),
)

LOWRANK_HC_RECIPES: tuple[str, ...] = (
    "hc_gate_mix",
    "hc_combine",
    "grouped_gemma_rmsnorm",
    "hc_combine_rmsnorm",
)
_ARCH35_IDS = frozenset({"npu_arch_3510", "dav_3510"})
FP32_WORKING_BYTES = 4
LOWRANK_HC_HIDDEN_TILE = 512
_MS_PER_SECOND = 1000.0
_NORM_OPS_PER_ELEMENT = 5
_SILU_OPS_PER_ELEMENT = 4
_MATRIX_FLOPS_PER_MAC = 2.0
_STANDARD_ROLES = frozenset({"mte2", "mte2_aiv", "mte2_aic", "mte3", "hbm", "vector"})


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


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _hc_grid_dims(op: OpSpec) -> tuple[int, int]:
    """Resolve ``(hc_mult, hidden_dim)`` from declared op metadata and shape."""
    meta = dict(op.metadata)
    view: dict = {}
    for subgraph in meta.get("structure_subgraphs", ()) or ():
        if isinstance(subgraph, dict) and isinstance(subgraph.get("tensor_view"), dict):
            view = subgraph["tensor_view"]
            break
    raw_hc = view.get("hc") or meta.get("hc_mult") or meta.get("num_groups") or meta.get("hc") or 1
    hc_mult = max(1, int(raw_hc))
    raw_hidden = view.get("hidden") or meta.get("hidden") or meta.get("hidden_size") or 0
    if raw_hidden:
        hidden = max(1, int(raw_hidden))
    elif len(op.local_shape) >= 2:
        width = max(1, int(op.local_shape[-1]))
        hidden = max(1, width // hc_mult) if op.recipe != "hc_gate_mix" else width
    else:
        hidden = 1
    return hc_mult, hidden


def _is_row_reduction_norm(op: OpSpec) -> bool:
    """True for full-hidden per-stream norm reductions (incl. unit-offset RMSNorm)."""
    meta = dict(op.metadata)
    norm_style = str(meta.get("norm_style", "") or "").strip().lower()
    return (
        str(op.family or "").strip().lower() == "rmsnorm"
        or norm_style == "unit_offset"
        or bool(meta.get("norm_unit_offset", False))
        or str(op.recipe or "").endswith("_rmsnorm")
    )


def _tile_weights(op: OpSpec, rows: int) -> tuple[int, ...]:
    """Per-program active element counts matching upstream masked Triton tiles."""
    hc_mult, hidden = _hc_grid_dims(op)
    if _is_row_reduction_norm(op):
        return (hidden,) * (rows * hc_mult)
    if op.recipe in {"hc_gate_mix", "hc_combine"}:
        hidden_tiles = _ceil_div(hidden, LOWRANK_HC_HIDDEN_TILE)
        col_widths = tuple(
            min(LOWRANK_HC_HIDDEN_TILE, hidden - idx * LOWRANK_HC_HIDDEN_TILE)
            for idx in range(hidden_tiles)
        )
        repeats = rows * (hc_mult if op.recipe == "hc_combine" else 1)
        return col_widths * repeats
    return (1,) * rows


def _owner_shares(weights: tuple[int, ...], active_aiv: int) -> tuple[float, ...]:
    total = float(sum(weights))
    return tuple(
        float(sum(weights[aiv_idx::active_aiv])) / total
        for aiv_idx in range(active_aiv)
    )


def _conserved_work(op: OpSpec) -> Work:
    from mindie_sim.prediction.physical_recipe_base import work_from_op

    base = work_from_op(op)
    fallback = str(op.dtype or "bf16")
    if not base.matrix and float(op.flops or 0.0) > 0.0:
        return Work({fallback: float(op.flops)}, dict(base.resources))
    if "" in base.matrix:
        matrix = dict(base.matrix)
        unnamed = matrix.pop("")
        matrix[fallback] = matrix.get(fallback, 0.0) + unnamed
        return Work(matrix, dict(base.resources))
    return base


def _hc_lowrank_dim(op: OpSpec) -> int:
    meta = dict(op.metadata)
    for subgraph in meta.get("structure_subgraphs", ()) or ():
        if isinstance(subgraph, dict) and isinstance(subgraph.get("tensor_view"), dict):
            raw = subgraph["tensor_view"].get("hc_lowrank")
            if raw:
                return max(0, int(raw))
    return max(0, int(meta.get("hc_lowrank", 0) or 0))


def _aiv_stage_program(
    context,
    stage_name: str,
    stage_res: dict[str, float],
    *,
    aiv_units: int,
    active_aiv: int,
    shares: tuple[float, ...],
    setup_ms: float = 0.0,
    caps: dict[str, float] | None = None,
) -> ParallelProgram:
    roles = tuple(role for role in stage_res if role != "hbm")
    branches = []
    for aiv_idx in range(aiv_units):
        if aiv_idx >= active_aiv:
            branches.append(KernelProgram(()))
            continue
        share = shares[aiv_idx]
        owned = context.owned(aiv_idx, aiv_units, roles)
        res = {role: amount * share for role, amount in stage_res.items() if amount}
        body = Stage(
            stage_name,
            context.bind(Work(resources=res)),
            after=(f"{stage_name}_setup",) if setup_ms > 0.0 else (),
            placement=owned,
            caps=caps or {},
        )
        if setup_ms > 0.0:
            branches.append(KernelProgram((Stage(f"{stage_name}_setup", latency=setup_ms), body)))
        else:
            branches.append(KernelProgram((body,)))
    return ParallelProgram(tuple(branches))


def _aic_stage(
    context,
    name: str,
    matrix_work: dict[str, float],
    resources: dict[str, float],
    *,
    after: tuple[str, ...] = (),
) -> Stage:
    aic_units = max(1, int(getattr(context, "effective_aic_cores", 1) or 1))
    roles = (("cube",) if matrix_work else ()) + tuple(r for r in resources if r != "hbm")
    owned = {k: v[:aic_units] for k, v in context.owned(0, 1, roles).items()}
    return Stage(
        name,
        context.bind(Work(matrix=matrix_work, resources=resources)),
        after=after,
        placement=owned,
    )


def _split_hbm_weights(total_hbm: float, weights: tuple[float, ...]) -> tuple[float, ...]:
    if total_hbm <= 0.0:
        return (0.0,) * len(weights)
    denom = sum(weights)
    if denom <= 0.0:
        return (0.0,) * (len(weights) - 1) + (total_hbm,)
    allocated = [total_hbm * (w / denom) for w in weights[:-1]]
    allocated.append(max(0.0, total_hbm - sum(allocated)))
    return tuple(allocated)


def _build_fused_pre_chain(
    op: OpSpec, geometry: Geometry, base: Work, rows: int,
    aiv_units: int, active_aiv: int, shares: tuple[float, ...], programs: int,
) -> OperatorPlan:
    hc_mult, hidden = _hc_grid_dims(op)
    width = hc_mult * hidden
    elt = float(dtype_bytes(op.dtype or "bf16"))
    rank = _hc_lowrank_dim(op)
    split_mte2 = (
        "mte2_aiv" in (geometry.resource_units or {})
        and "mte2_aic" in (geometry.resource_units or {})
    )
    generic_read = float(base.resources.get("mte2", 0.0))
    explicit_aiv = float(base.resources.get("mte2_aiv", 0.0))
    explicit_aic = float(base.resources.get("mte2_aic", 0.0))
    write = float(base.resources.get("mte3", 0.0))
    hbm = float(base.resources.get("hbm", 0.0))
    vector = float(base.resources.get("vector", 0.0))
    extra = {k: float(v) for k, v in base.resources.items() if k not in _STANDARD_ROLES and v}

    norm_from_generic = min(generic_read, float(rows + 1) * float(width) * elt)
    proj_from_generic = generic_read - norm_from_generic
    aiv_read = explicit_aiv + norm_from_generic
    aic_read = explicit_aic + proj_from_generic

    plan_res = dict(base.resources)
    if split_mte2 and generic_read > 0.0:
        plan_res.pop("mte2", None)
        if aiv_read > 0.0:
            plan_res["mte2_aiv"] = aiv_read
        if aic_read > 0.0:
            plan_res["mte2_aic"] = aic_read
    plan_work = Work(dict(base.matrix), plan_res)

    total_flops = sum(base.matrix.values())
    if rank > 0 and total_flops > 0.0:
        up_frac = min(1.0, max(0.0, (_MATRIX_FLOPS_PER_MAC * rows * width * rank) / total_flops))
    else:
        up_frac = 0.5
    down_frac = 1.0 - up_frac
    down_matrix = {k: v * down_frac for k, v in base.matrix.items() if v * down_frac > 0.0}
    up_matrix = {k: v - down_matrix.get(k, 0.0) for k, v in base.matrix.items() if v - down_matrix.get(k, 0.0) > 0.0}
    down_read = aic_read * down_frac
    up_read = aic_read - down_read

    norm_vec = min(vector, float(_NORM_OPS_PER_ELEMENT * rows * width))
    silu_vec = min(vector - norm_vec, float(_SILU_OPS_PER_ELEMENT * rows * rank)) if rank > 0 else 0.0
    mix_vec = vector - norm_vec - silu_vec
    hbm_norm, hbm_down, hbm_up, hbm_mix = _split_hbm_weights(hbm, (aiv_read, down_read, up_read, write))

    def program(context):
        aiv_r = "mte2_aiv" if "mte2_aiv" in context.resource_names else "mte2"
        aic_r = "mte2_aic" if "mte2_aic" in context.resource_names else "mte2"
        stages: list[Stage] = []
        prev: tuple[str, ...] = ()

        norm_res = {k: v for k, v in ((aiv_r, aiv_read), ("hbm", hbm_norm), ("vector", norm_vec)) if v}
        if norm_res:
            stages.append(Stage(
                "hc_ub_norm",
                program=_aiv_stage_program(
                    context, "hc_ub_norm_body", norm_res,
                    aiv_units=aiv_units, active_aiv=active_aiv, shares=shares,
                ),
            ))
            prev = ("hc_ub_norm",)

        down_res = {k: v for k, v in ((aic_r, down_read), ("hbm", hbm_down)) if v}
        if down_matrix or down_res:
            stages.append(_aic_stage(context, "hc_ub_down_proj", down_matrix, down_res, after=prev))
            prev = ("hc_ub_down_proj",)

        if silu_vec > 0.0:
            stages.append(Stage(
                "hc_ub_silu", after=prev,
                program=_aiv_stage_program(
                    context, "hc_ub_silu_body", {"vector": silu_vec},
                    aiv_units=aiv_units, active_aiv=active_aiv, shares=shares,
                ),
            ))
            prev = ("hc_ub_silu",)

        up_res = {k: v for k, v in ((aic_r, up_read), ("hbm", hbm_up)) if v}
        if up_matrix or up_res:
            stages.append(_aic_stage(context, "hc_ub_up_proj", up_matrix, up_res, after=prev))
            prev = ("hc_ub_up_proj",)

        mix_res = {k: v for k, v in (("vector", mix_vec), ("mte3", write), ("hbm", hbm_mix)) if v}
        mix_res.update(extra)
        if mix_res:
            stages.append(Stage(
                "hc_ub_gate_mix", after=prev,
                program=_aiv_stage_program(
                    context, "hc_ub_gate_mix_body", mix_res,
                    aiv_units=aiv_units, active_aiv=active_aiv, shares=shares,
                ),
            ))
        return KernelProgram(tuple(stages))

    return OperatorPlan(plan_work, program, {
        "execution_class": "vector_aiv",
        "hc_variant": str(op.recipe),
        "fused_ub_resident": True,
        "logical_programs": programs,
        "ub_compute_dtype": "fp32",
        "ub_working_bytes_per_elem": FP32_WORKING_BYTES,
    })


def _build_split_stage_program(
    context,
    base: Work,
    *,
    aiv_units: int,
    active_aiv: int,
    shares: tuple[float, ...],
    stage_prefix: str,
    setup_ms: float = 0.0,
    caps: dict[str, float] | None = None,
):
    generic_read = float(base.resources.get("mte2", 0.0))
    read_aiv = float(base.resources.get("mte2_aiv", 0.0))
    read_aic = float(base.resources.get("mte2_aic", 0.0))
    write = float(base.resources.get("mte3", 0.0))
    hbm = float(base.resources.get("hbm", 0.0))
    vector = float(base.resources.get("vector", 0.0))
    extra = {k: float(v) for k, v in base.resources.items() if k not in _STANDARD_ROLES and v}
    matrix_work = dict(base.matrix)

    read_pipe = context.read_pipe() if generic_read > 0.0 else "mte2"
    generic_on_cube = bool(matrix_work) and generic_read > 0.0 and read_pipe == "mte2_aic"
    cube_res: dict[str, float] = {}
    if generic_on_cube:
        cube_res["mte2"] = generic_read
    if read_aic > 0.0:
        cube_res["mte2_aic"] = read_aic

    aiv_res: dict[str, float] = {}
    if hbm > 0.0:
        aiv_res["hbm"] = hbm
    if generic_read > 0.0 and not generic_on_cube:
        aiv_res["mte2"] = generic_read
    if read_aiv > 0.0:
        aiv_res["mte2_aiv"] = read_aiv
    if vector > 0.0:
        aiv_res["vector"] = vector
    if write > 0.0:
        aiv_res["mte3"] = write
    aiv_res.update(extra)

    if matrix_work or cube_res:
        cube_stage = _aic_stage(context, f"{stage_prefix}_cube", matrix_work, cube_res)
        if not aiv_res:
            return KernelProgram((cube_stage,))
        aiv_prog = _aiv_stage_program(
            context, f"{stage_prefix}_body", aiv_res,
            aiv_units=aiv_units, active_aiv=active_aiv, shares=shares,
            setup_ms=setup_ms, caps=caps,
        )
        return KernelProgram((
            cube_stage,
            Stage(f"{stage_prefix}_aiv", after=(cube_stage.name,), program=aiv_prog),
        ))
    return _aiv_stage_program(
        context, f"{stage_prefix}_body", aiv_res,
        aiv_units=aiv_units, active_aiv=active_aiv, shares=shares,
        setup_ms=setup_ms, caps=caps,
    )


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
    aiv_units = max(1, int((geometry.resource_units or {}).get("vector", 1)))
    weights = _tile_weights(op, rows)
    programs = len(weights)
    active_aiv = min(aiv_units, programs)
    shares = _owner_shares(weights, active_aiv)
    if bool(dict(op.metadata).get("mhc_lowrank_fused_pre", False)) and bool(base.matrix):
        return _build_fused_pre_chain(op, geometry, base, rows, aiv_units, active_aiv, shares, programs)

    def program(context):
        return _build_split_stage_program(
            context, base,
            aiv_units=aiv_units, active_aiv=active_aiv, shares=shares,
            stage_prefix="hc_ub",
        )

    return OperatorPlan(base, program, {
        "execution_class": "vector_aiv",
        "hc_variant": str(op.recipe),
        "fused_ub_resident": True,
        "logical_programs": programs,
        "ub_compute_dtype": "fp32",
        "ub_working_bytes_per_elem": FP32_WORKING_BYTES,
    })


def build_lowrank_hc_vector(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Build an Arch35 strided multi-stream HC vector plan conserving ``work_from_op``."""
    from mindie_sim.prediction.physical_recipe_base import build_plan
    from mindie_sim.prediction.operators.vector.vector_pipeline import AIVTiledVectorAndGatherRecipe

    base = _conserved_work(op)
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
    aiv_units = max(1, int((geometry.resource_units or {}).get("vector", 1)))
    weights = _tile_weights(op, rows)
    programs = len(weights)
    active_aiv = min(aiv_units, programs)
    shares = _owner_shares(weights, active_aiv)
    concurrent_share = 1.0 / float(active_aiv)

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
                caps[hbm_phys] = (float(strided_bps) / _MS_PER_SECOND) * concurrent_share
        setup_ms = 0.0
        if tbl is not None and dp is not None:
            env = resolve_dispatch_envelope(tbl, dp)
            if env is not None:
                setup_ms = float(env.aiv_floor_ms(active_aiv))
        return _build_split_stage_program(
            context, base,
            aiv_units=aiv_units, active_aiv=active_aiv, shares=shares,
            stage_prefix="hc_strided", setup_ms=setup_ms, caps=caps,
        )

    return OperatorPlan(base, program, {
        "execution_class": "vector_aiv",
        "hc_variant": str(op.recipe),
        "logical_programs": programs,
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
