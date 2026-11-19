"""CANN 9.3 Arch35 mHC, ``text_gated_residual``, and TP ``MC2`` V2 recipes.

[Role]
  Model ``_C_ascend.npu_hc_pre_v3`` / ``npu_hc_pre_v2`` (dynamic tiling across
  ``C in {2560, 4096, 5120, 7168}`` + BF16->FP32 norm cast), ``#22142``
  ``mhc_post_collapse_norm`` (``<= 72`` rows), ``#21794``
  ``cann_ops_transformer.text_gated_residual`` (Low-Rank Gated HC native op),
  and ``#22385`` / ``#21832`` ``AllGatherMatmulV2`` & ``MatmulReduceScatterV2``
  TP communication + Cube overlap recipes.

[Invariants]
  * ``OperatorPlan.work`` equals the exact sum of issued per-owner ``Stage``
    demands across ``hbm``, ``l2``, ``mte2_aic``, ``mte2_aiv``, ``mte1``,
    ``cube``, ``vector``, ``fixpipe``, ``mte3``, and ``link``.
  * ``mhc_post_collapse_norm`` enforces ``rows <= 72`` unless ``unbounded=True``.
  * Zero model-name branches or ``ResourceEnvelope`` fallbacks participate.

[Timeouts & Locks]
  O(compute_units) plan construction; bounded finite stages, no I/O or locks.

[Side-Effect Policy]
  Pure immutable ``OpSpec`` and ``OperatorPlan`` construction.
"""
from __future__ import annotations

import math
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from ..mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from ..ownership import read_role, unit_group
from ..program import KernelProgram, ParallelProgram, Stage

HC_PRE_V3_RECIPE = "npu_hc_pre_v3_arch35"
HC_PRE_V2_RECIPE = "npu_hc_pre_v2_arch35"
MHC_POST_COLLAPSE_RECIPE = "mhc_post_collapse_norm_arch35"
TEXT_GATED_RESIDUAL_RECIPE = "text_gated_residual_arch35"
ALL_GATHER_MATMUL_V2_RECIPE = "all_gather_matmul_v2_arch35"
MATMUL_REDUCE_SCATTER_V2_RECIPE = "matmul_reduce_scatter_v2_arch35"

HC_PRE_TILING_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "csrc/moe/hc_pre/op_host/hc_pre_tiling_arch35.h",
    "source_sha256": "9ffce3d0faf8c1f0c54a64fdb6e11d2f0476e7a99703a418e1a112f07196c8b0",
}
DSV41_MODEL_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/models/deepseek_v41/model.py",
    "source_sha256": "1006ded6d3c5b394299bdabb7619752198adbd6ab3612c058ef3659bbf9e24c0",
}
PR22142_MHC_SOURCE = {
    "repository": "vllm-project/vllm-ascend", "revision": "09d2eaea4d",
    "relative_path": "vllm_ascend/models/deepseek_v41/model.py",
    "source_sha256": "1006ded6d3c5b394299bdabb7619752198adbd6ab3612c058ef3659bbf9e24c0",
}
TEXT_GATED_RESIDUAL_SOURCE = {
    "repository": "cann_ops_transformer", "revision": "1.0.0",
    "relative_path": "cann_ops_transformer/ops/mhc/text_gated_residual/text_gated_residual.py",
    "source_sha256": "8c3fde722b28617b1eebd12411ee19358cf2c1bb529e9e58741043f39318c9ca",
}
TEXT_GATED_HEADER_SOURCE = {
    "repository": "cann-toolkit", "revision": "9.3.0-20261112_192141454",
    "relative_path": "include/aclnnop/aclnn_text_gated_residual.h",
    "source_sha256": "804edc6ba4f1403c04cd1eaef30a65d13658a01a773c43b41aa8cb5cfd4748dd",
}
AG_MM_V2_SOURCE = {
    "repository": "cann-toolkit", "revision": "9.3.0-20261112_192141454",
    "relative_path": "include/aclnnop/aclnn_all_gather_matmul_v2.h",
    "source_sha256": "0bc5be4511f72e8c2a232b8f1c0d48292bfc02606a412e2fd0fd6cdd41b6e267",
}
MM_RS_V2_SOURCE = {
    "repository": "cann-toolkit", "revision": "9.3.0-20261112_192141454",
    "relative_path": "include/aclnnop/aclnn_matmul_reduce_scatter_v2.h",
    "source_sha256": "8bbd520d9e46c6a574c88f4338ee324190c53b0f3747e03ad9a79456a14c3c44",
}

BF16_BYTES = 2
FP32_BYTES = 4
MX_GROUP_SIZE = 32
DEFAULT_HC_STREAMS = 4
DEFAULT_LOWRANK = 16
DEFAULT_HC_HIDDEN = 5120
PR22142_ROW_CUTOFF = 72
HC_WIDTH_ALIGNMENT = 128
SINKHORN_ITERS = 4
SINKHORN_OPS_PER_ELEM = 8
RMSNORM_OPS_PER_ELEM = 5
GATED_RES_OPS_PER_ELEM = 7
RS_REDUCE_OPS_PER_ELEM = 2
MIN_HC_UB_BYTES = 32 * 1024
MC2_L1_BYTES = 192 * 1024
L2_EFFECTIVE_SHARE = 0.75


def _effective_l2(geometry: Geometry) -> float:
    mechanisms = getattr(geometry, "physical_mechanisms", None)
    hierarchy = getattr(mechanisms, "memory_hierarchy", None)
    if hierarchy is not None:
        return float(hierarchy.effective_l2_bytes)
    return L2_EFFECTIVE_SHARE * float((geometry.buffers or {}).get("l2", 0))


def make_hc_op(
    *, variant: str, op_id: str, deps: tuple[str, ...] = (), stage: int = 0,
    layer: int = 0, requests: tuple[str, ...] = (), phase: str = "decode",
    rows: int, hidden_size: int, hc_mult: int = DEFAULT_HC_STREAMS,
    hc_lowrank: int = DEFAULT_LOWRANK, unbounded_rows: bool = False,
) -> OpSpec:
    """Construct an mHC (``hc_pre_v2/v3``, ``mhc_post_collapse_norm``) or ``text_gated_residual`` op."""
    tok, width = max(1, int(rows)), max(1, int(hidden_size))
    streams, lowrank = max(1, int(hc_mult)), max(1, int(hc_lowrank))
    if width % HC_WIDTH_ALIGNMENT != 0:
        raise ModelingError(f"hc_arch35_width_unaligned:{width}")
    kind = str(variant).strip().lower()
    if kind == "mhc_post_collapse" and not unbounded_rows and tok > PR22142_ROW_CUTOFF:
        raise ModelingError(f"mhc_post_collapse_row_cutoff_exceeded:{tok}")

    recipe_map = {
        "hc_pre_v3": (HC_PRE_V3_RECIPE, "npu_hc_pre_v3", HC_PRE_TILING_SOURCE),
        "hc_pre_v2": (HC_PRE_V2_RECIPE, "npu_hc_pre_v2", HC_PRE_TILING_SOURCE),
        "mhc_post_collapse": (MHC_POST_COLLAPSE_RECIPE, "mhc_post_collapse_norm", PR22142_MHC_SOURCE),
        "text_gated_residual": (TEXT_GATED_RESIDUAL_RECIPE, "aclnnTextGatedResidual", TEXT_GATED_RESIDUAL_SOURCE),
    }
    if kind not in recipe_map:
        raise ModelingError(f"unsupported_hc_variant:{variant}")
    recipe, native_kind, src = recipe_map[kind]
    flops = (
        float(2 * tok * (streams * width) * (2 * lowrank + streams))
        if kind == "text_gated_residual"
        else float(2 * tok * (streams * width) * ((2 + streams) * streams))
    )
    shape = (tok, streams, width)
    meta = {
        "native_kind": native_kind,
        "hc_geometry": {
            "variant": kind, "rows": tok, "hidden_size": width,
            "hc_mult": streams, "hc_lowrank": lowrank,
            "unbounded_rows": bool(unbounded_rows),
        },
        "native_source": src,
        "header_source": TEXT_GATED_HEADER_SOURCE if kind == "text_gated_residual" else DSV41_MODEL_SOURCE,
    }
    return OpSpec(
        op_id=op_id, deps=deps, family="mhc", recipe=recipe,
        phase=phase, stage=stage, layer=layer, requests=requests,
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype="bf16", flops=flops, matrix_flops=(("bf16", flops),),
        source_boundary="kernel", metadata=tuple(sorted(meta.items())),
    )


def make_tp_mc2_op(
    *, direction: str, op_id: str, deps: tuple[str, ...] = (), stage: int = 0,
    layer: int = 0, requests: tuple[str, ...] = (), phase: str = "decode",
    global_tokens: int, n_dim: int, k_dim: int, tp: int,
    compute_dtype: str = "fp8_e4m3",
) -> OpSpec:
    """Construct ``AllGatherMatmulV2`` or ``MatmulReduceScatterV2`` (#22385) op."""
    mode = str(direction).strip().lower()
    if mode not in {"all_gather_matmul", "matmul_reduce_scatter"}:
        raise ModelingError(f"unsupported_tp_mc2_direction:{direction}")
    tok, n_out, k_in, tp_deg = max(1, int(global_tokens)), max(1, int(n_dim)), max(1, int(k_dim)), max(1, int(tp))
    is_ag = mode == "all_gather_matmul"
    recipe = ALL_GATHER_MATMUL_V2_RECIPE if is_ag else MATMUL_REDUCE_SCATTER_V2_RECIPE
    src = AG_MM_V2_SOURCE if is_ag else MM_RS_V2_SOURCE
    flops = float(2 * tok * n_out * k_in)
    comm_width = k_in if is_ag else n_out
    elem_bytes = 1.0 + 1.0 / MX_GROUP_SIZE if (is_ag and compute_dtype == "fp8_e4m3") else float(BF16_BYTES)
    link_bytes = float(tok * comm_width * elem_bytes * (tp_deg - 1) / tp_deg) if tp_deg > 1 else 0.0
    shape = (tok if is_ag else max(1, math.ceil(tok / tp_deg)), n_out)
    meta = {
        "native_kind": "aclnnAllGatherMatmulV2" if is_ag else "aclnnMatmulReduceScatterV2",
        "tp_mc2_geometry": {
            "direction": mode, "global_tokens": tok, "n_dim": n_out,
            "k_dim": k_in, "tp": tp_deg, "compute_dtype": compute_dtype,
            "link_bytes": link_bytes,
        },
        "native_source": src,
    }
    return OpSpec(
        op_id=op_id, deps=deps, family="gemm", recipe=recipe,
        phase=phase, stage=stage, layer=layer, requests=requests,
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype="bf16", flops=flops, matrix_flops=((compute_dtype, flops),),
        link_bytes=link_bytes, source_boundary="kernel",
        metadata=tuple(sorted(meta.items())),
    )


def build_hc(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Build a work-conserving AIC+AIV program for mHC or ``text_gated_residual``."""
    ub_cap = int((geometry.buffers or {}).get("ub", 0))
    if ub_cap > 0 and ub_cap < MIN_HC_UB_BYTES:
        raise ModelingError("hc_arch35_ub_capacity_insufficient")
    meta = dict(op.metadata)
    hg = dict(meta.get("hc_geometry") or {})
    kind = str(hg.get("variant", "hc_pre_v3"))
    tok = max(1, int(hg.get("rows", op.local_shape[0] if op.local_shape else 1)))
    streams = max(1, int(hg.get("hc_mult", DEFAULT_HC_STREAMS)))
    width = max(1, int(hg.get("hidden_size", op.local_shape[-1] if op.local_shape else DEFAULT_HC_HIDDEN)))
    lowrank = max(1, int(hg.get("hc_lowrank", DEFAULT_LOWRANK)))

    hc_dim = streams * width
    if kind == "text_gated_residual":
        w_bytes = float((hc_dim * (2 * lowrank + streams) + hc_dim) * BF16_BYTES)
        flops = float(2 * tok * hc_dim * (2 * lowrank + streams))
        vec_ops = float(tok * hc_dim * (RMSNORM_OPS_PER_ELEM + GATED_RES_OPS_PER_ELEM))
    else:
        mix = (2 + streams) * streams
        w_bytes = float(hc_dim * mix * BF16_BYTES + mix * FP32_BYTES + hc_dim * BF16_BYTES)
        flops = float(2 * tok * hc_dim * mix)
        cast_ops = 1 if kind in {"hc_pre_v3", "hc_pre_v2"} else 0
        vec_ops = float(tok * hc_dim * (RMSNORM_OPS_PER_ELEM + cast_ops) + tok * mix * SINKHORN_ITERS * SINKHORN_OPS_PER_ELEM)
    out_write = float(tok * (hc_dim + width) * BF16_BYTES)
    aic_read = float(tok * hc_dim * BF16_BYTES) + w_bytes

    has_l2 = bool((geometry.rates or {}).get("l2", 0.0) or (geometry.resource_units or {}).get("l2", 0))
    l2_cap = _effective_l2(geometry) if has_l2 else 0.0
    l2_hit = min(w_bytes, l2_cap) if l2_cap > 0.0 else 0.0
    aic_gm_read = max(0.0, aic_read - l2_hit)
    fix_bytes = float(tok * hc_dim * BF16_BYTES)

    cores = max(1, int(geometry.compute_units))
    aiv_units = max(1, int((geometry.resource_units or {}).get("vector", 2 * cores)))
    aic_pipe = read_role(geometry.resource_units or {}, aiv=None)
    plan_res: dict[str, float] = {
        "hbm": aic_gm_read + out_write, aic_pipe: aic_read,
        "mte1": aic_read, "fixpipe": fix_bytes,
        "vector": vec_ops, "mte3": out_write,
    }
    if l2_hit > 0.0:
        plan_res["l2"] = l2_hit
    total_work = Work(matrix={"bf16": flops}, resources=plan_res)
    ub_alloc = min(ub_cap, MIN_HC_UB_BYTES) if ub_cap > 0 else 0

    def program(context):
        c_aic = read_role(context.resource_names, aiv=None)
        share = 1.0 / float(cores)

        def per_core(core: int):
            aiv_idx = min(aiv_units - 1, 2 * core if aiv_units >= 2 * cores else core)
            aic_owned = context.owned(core, cores, ("cube", "mte1", "fixpipe", c_aic))
            aiv_owned = context.owned(aiv_idx, aiv_units, ("vector", "mte3"))
            alloc = (
                (context.allocate(
                    f"hc_ub_{core}", "ub", ub_alloc,
                    unit_group(aiv_idx, aiv_units, context.pool_units.get("ub", aiv_units)),
                ),)
                if ("ub" in context.pool_names and ub_alloc > 0) else ()
            )
            aic_res: dict[str, float] = {
                "hbm": aic_gm_read * share, c_aic: aic_read * share,
                "mte1": aic_read * share, "fixpipe": fix_bytes * share,
            }
            if l2_hit > 0.0:
                aic_res["l2"] = l2_hit * share
            s_proj = Stage("hc_lowrank_mix", context.bind(Work(
                matrix={"bf16": flops * share}, resources=aic_res)),
                acquire=alloc, placement=aic_owned)
            s_gate = Stage("hc_norm_gate_store", context.bind(Work(resources={
                "vector": vec_ops * share, "hbm": out_write * share, "mte3": out_write * share,
            })), after=("hc_lowrank_mix",), free=alloc, placement=aiv_owned)
            return KernelProgram((s_proj, s_gate))

        return ParallelProgram(tuple(per_core(c) for c in range(cores)))

    return OperatorPlan(total_work, program, {
        "execution_class": "mixed_aic_aiv", "hc_variant": kind,
        "source_program": meta.get("native_source") or HC_PRE_TILING_SOURCE,
    })


def build_tp_mc2(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Build a comm-overlapped AIC+Link program for ``AllGatherMatmulV2`` / ``MatmulReduceScatterV2``."""
    l1_cap = int((geometry.buffers or {}).get("l1", 0))
    if l1_cap > 0 and l1_cap < MC2_L1_BYTES:
        raise ModelingError("tp_mc2_v2_l1_capacity_insufficient")
    meta = dict(op.metadata)
    tg = dict(meta.get("tp_mc2_geometry") or {})
    mode = str(tg.get("direction", "all_gather_matmul"))
    is_ag = mode == "all_gather_matmul"
    tok = max(1, int(tg.get("global_tokens", op.local_shape[0] if op.local_shape else 1)))
    n_out = max(1, int(tg.get("n_dim", op.local_shape[-1] if op.local_shape else 1024)))
    k_in = max(1, int(tg.get("k_dim", n_out)))
    tp_deg = max(1, int(tg.get("tp", 2)))
    comp_dt = str(tg.get("compute_dtype", "fp8_e4m3"))
    has_link = bool((geometry.rates or {}).get("link", 0.0) or (geometry.resource_units or {}).get("link", 0))
    link_bytes = float(tg.get("link_bytes", op.link_bytes or 0.0)) if has_link else 0.0

    elem_a = 1.0 + 1.0 / MX_GROUP_SIZE if comp_dt == "fp8_e4m3" else float(BF16_BYTES)
    aic_read = float((tok + n_out) * k_in * elem_a)
    out_rows = tok if is_ag else max(1, math.ceil(tok / tp_deg))
    out_write = float(out_rows * n_out * BF16_BYTES)
    flops = float(2 * tok * n_out * k_in)
    vec_ops = 0.0 if is_ag else float(out_rows * n_out * RS_REDUCE_OPS_PER_ELEM)
    fix_bytes = float(tok * n_out * BF16_BYTES)

    cores = max(1, int(geometry.compute_units))
    aiv_units = max(1, int((geometry.resource_units or {}).get("vector", 2 * cores)))
    aic_pipe = read_role(geometry.resource_units or {}, aiv=None)
    plan_res: dict[str, float] = {
        "hbm": aic_read + out_write, aic_pipe: aic_read,
        "mte1": aic_read, "fixpipe": fix_bytes,
    }
    if not is_ag:
        plan_res["vector"] = vec_ops
        plan_res["mte3"] = out_write
    if link_bytes > 0.0:
        plan_res["link"] = link_bytes
    total_work = Work(matrix={comp_dt: flops}, resources=plan_res)

    def program(context):
        c_aic = read_role(context.resource_names, aiv=None)
        share = 1.0 / float(cores)

        def per_core(core: int):
            aiv_idx = min(aiv_units - 1, 2 * core if aiv_units >= 2 * cores else core)
            aic_owned = context.owned(core, cores, ("cube", "mte1", "fixpipe", c_aic))
            aiv_owned = context.owned(aiv_idx, aiv_units, ("vector", "mte3"))
            if is_ag:
                comm_s = Stage(
                    "ag_comm_stream",
                    context.bind(Work(resources={"link": link_bytes * share} if link_bytes > 0.0 else {})),
                    keep_empty=link_bytes <= 0.0,
                )
                mm_s = Stage("overlapped_matmul", context.bind(Work(
                    matrix={comp_dt: flops * share},
                    resources={
                        "hbm": (aic_read + out_write) * share, c_aic: aic_read * share,
                        "mte1": aic_read * share, "fixpipe": fix_bytes * share,
                    },
                )), placement=aic_owned)
                return KernelProgram((comm_s, mm_s))
            mm_s = Stage("tiled_matmul", context.bind(Work(
                matrix={comp_dt: flops * share},
                resources={
                    "hbm": aic_read * share, c_aic: aic_read * share,
                    "mte1": aic_read * share, "fixpipe": fix_bytes * share,
                },
            )), placement=aic_owned)
            rs_res: dict[str, float] = {
                "vector": vec_ops * share, "hbm": out_write * share, "mte3": out_write * share,
            }
            if link_bytes > 0.0 and "link" in context.resource_names:
                rs_res["link"] = link_bytes * share
            rs_s = Stage("overlapped_rs", context.bind(Work(resources=rs_res)), placement=aiv_owned)
            return KernelProgram((mm_s, rs_s))

        return ParallelProgram(tuple(per_core(c) for c in range(cores)))

    return OperatorPlan(total_work, program, {
        "execution_class": "comm_overlapped_cube", "direction": mode,
        "source_program": meta.get("native_source") or AG_MM_V2_SOURCE,
    })


def recipes() -> dict[str, PlannedRecipe]:
    hc_plan = PlannedRecipe(build_hc)
    mc2_plan = PlannedRecipe(build_tp_mc2)
    return {
        HC_PRE_V3_RECIPE: hc_plan, HC_PRE_V2_RECIPE: hc_plan,
        MHC_POST_COLLAPSE_RECIPE: hc_plan, TEXT_GATED_RESIDUAL_RECIPE: hc_plan,
        ALL_GATHER_MATMUL_V2_RECIPE: mc2_plan, MATMUL_REDUCE_SCATTER_V2_RECIPE: mc2_plan,
    }
