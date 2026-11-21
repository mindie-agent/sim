"""Constructive CATLASS and DSL candidate search.

[Role] Translate declared matrix and tile-pipeline contracts into replay candidates.
[Invariants] Explicit pool ownership, supported routes and full verified seed retention.
[Timeouts & Locks] Candidate and oracle trial counts bound search, with no search locks.
[Side-Effect Policy] Read-only capability queries and oracle replays.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from importlib import import_module
from math import prod
from types import SimpleNamespace
from typing import TYPE_CHECKING

from mindie_sim.prediction.runtime.tile_solver import (
    AIC_POOLS, AIV_POOLS, LiveBufferSpec, TileCandidate,
    TileInfeasibleError, TilePipelineSpec, TileShape, solve as solve_tiles,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import DatapathCapabilitySpec, datapath_for_die
from mindie_sim.schema.operators import OpSpec

from .contracts import CatlassKernelSpec, DslKernelSpec, FuseRegionSpec
from .dual_bound import _unique_ops
from .framework_solver import (
    DEFAULT_MAX_TRIALS, FrameworkSolveResult, _TrialSearch, _atomic_ops, _machine,
    _resolved, _stream_candidates, _stream_plan, _wall, _with_operations,
)
from .reload_bounds import BITS_PER_BYTE

if TYPE_CHECKING:
    from mindie_sim.rewrite.oracle import FullPlanEvaluation, ScenarioDossier
    from mindie_sim.rewrite.schema import RewritePlan

MATRIX_RANK = 3
ATTN_RES_WIDTH = 7168
ATTN_RES_MAX_TILE_ROWS = 4095
ATTN_RES_STAT_GRANULE = 8
ATTN_RES_STAT_BYTES = 12
BF16_COL_GRANULE = 16
FP32_ELEMENT_BYTES = 4
SWIZZLE_OFFSETS = (1, 2, 3, 4, 6, 8, 12, 16)
MUTEX_LIMIT = 28
MATRIX_BLOCK = 64
L1_K_TILE = 128
MX_SCALE_GROUP = 32
A8W4_SCALE_FACTOR = 16
CATLASS_SIMT_THREADS = 1024
TLA_SIMT_THREADS = 2048
SIMT_UB_CARVEOUT = 16 * 1024
MX_FORMATS = frozenset({"mxfp8", "mxfp4", "a8w4_mx"})
MATRIX_FAMILIES = frozenset({"gemm", "matmul", "linear", "grouped_gemm", "grouped_matmul"})
GROUPED_FAMILIES = frozenset({"grouped_gemm", "grouped_matmul", "grouped_moe", "mc2"})
FORMAT_BITS = {"fp32": 32, "float32": 32, "int32": 32, "bf16": 16, "bfloat16": 16,
               "fp16": 16, "float16": 16, "fp8": 8, "mxfp8": 8, "mxfp4": 4,
               "int8": 8, "int4": 4, "fp4": 4}
DSL_KERNELS = {
    "flash_kda": "mix_cube_vector",
    "mega_recurrent_kda": "mix_cube_vector",
    "commit_recurrent_kda": "vector_only",
    "mla_prolog_v4": "mix_cube_vector",
    "block_attn_res_update_rms_norm": "vector_only",
    "quant_sparse_lightning_indexer": "mix_cube_vector",
    "mixed_quant_sparse_flash_mla": "mix_cube_vector",
}
DSL_STAGES = ("D1_dsl_fuse", "D2_catlass", "D3_retune")
__all__ = ("DslSolveResult", "streamk_crossing_partials", "select_catlass_candidates",
           "select_dsl_candidates", "solve_dsl")

@dataclass(frozen=True, slots=True)
class DslSolveResult:
    best_plan: RewritePlan
    best_eval: FullPlanEvaluation
    best_wall_ns: int
    catlass_specs: tuple[CatlassKernelSpec, ...]
    dsl_specs: tuple[DslKernelSpec, ...]
    fuse_specs: tuple[FuseRegionSpec, ...]
    stage_history: tuple[tuple[str, int, str], ...]
    tried_candidates: int

def _ceil_div(numerator, denominator):
    return (numerator + denominator - 1) // denominator

def streamk_crossing_partials(b_tot: int, g_k: int, c_aic: int) -> int:
    if b_tot < 0 or g_k < 1 or c_aic < 0:
        raise ValueError("invalid StreamK geometry")
    if not c_aic:
        return 0
    iters = b_tot % c_aic * g_k
    base, extra = divmod(iters, c_aic)
    partials = start = 0
    for core in range(min(iters, c_aic)):
        end = start + base + (core < extra)
        partials += (end - 1) // g_k - start // g_k + 1
        start = end
    return partials

def _external_selector(catalog, selector):
    module_name = f"mindie_sim.templates.{catalog}"
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name not in {"mindie_sim.templates", module_name}:
            raise
        return None
    return getattr(module, selector, None)

def _bits(datapath, dtype):
    declared = datapath.compute_arch.declared
    if declared is not None:
        try:
            return declared.representation(dtype).bits
        except ModelingError:
            pass
    return FORMAT_BITS.get(dtype, 0)

def _payload(elements, bits):
    return _ceil_div(elements * bits, BITS_PER_BYTE)

def _checked_tiles(pipeline, datapath, aic, aiv, op=None):
    candidates = []
    for candidate in pipeline.candidates:
        if (candidate.owner_axis == "aic" and not aic) or (
            candidate.owner_axis == "aiv" and not aiv
        ):
            continue
        for buf in candidate.buffers:
            if ((buf.pool in AIC_POOLS and buf.owner_kind != "aic") or
                    (buf.pool in AIV_POOLS and buf.owner_kind != "aiv") or
                    (buf.pool == "l2" and buf.owner_kind not in {"aic", "aiv", "die"})):
                raise ModelingError(f"live_buffer_pool_owner_mismatch:{buf.name}")
        if any((buf.owner_kind == "aic" and not aic) or
               (buf.owner_kind == "aiv" and not aiv) for buf in candidate.buffers):
            continue
        candidates.append(candidate)
    # This worktree's runtime accepts explicit owners but does not yet reject
    # mislabelled buffers itself. Apply the pool-owner check before calling it.
    hw = SimpleNamespace(buffers=datapath.buffers, datapath=datapath,
                         stream_owners=SimpleNamespace(aic_limit=aic, aiv_limit=aiv))
    carveout = max(pipeline.simt_ub_carveout_bytes, datapath.simt_ub_carveout_bytes)
    checked = replace(pipeline, candidates=tuple(candidates), simt_ub_carveout_bytes=carveout)
    try:
        return solve_tiles(checked, op, hw)
    except TileInfeasibleError:
        return None

def _matrix_pipeline(spec, op, datapath, shape):
    m, n, k = shape
    lm, ln, lk = spec.l1_tile_shape
    tm, tn, tk = spec.l0_tile_shape
    a_bits, b_bits, acc_bits = (_bits(datapath, dtype)
                               for dtype in (spec.dtype_a, spec.dtype_b, spec.dtype_acc))
    if not min(a_bits, b_bits, acc_bits):
        return None
    b_input_bits = b_bits
    if spec.prologue_b == "DecompressFp4ToFp8":
        b_bits = FORMAT_BITS["fp8"]
    output_blocks = _ceil_div(m, tm) * _ceil_div(n, tn)
    if spec.scheduler == "grouped_slice_m":
        rows = tuple(dict(op.metadata).get("group_rows", ()))
        if rows:
            output_blocks = sum(_ceil_div(count, tm) for count in rows) * _ceil_div(n, tn)
    buffers = []
    is_simt = spec.simt_thread_block_dim is not None
    if not is_simt:
        for index, (name, pool, elements, bits) in enumerate((
            ("a_l1", "l1", lm * lk, a_bits), ("b_l1", "l1", ln * lk, b_bits),
            ("a_l0", "l0a", tm * tk, a_bits), ("b_l0", "l0b", tn * tk, b_bits),
            ("acc_l0", "l0c", tm * tn, acc_bits),
        )):
            scale = 0
            if pool == "l1" and spec.mx_scale_group:
                scale = _ceil_div(elements, spec.mx_scale_group) * spec.l1_scale_factor_k
            buffers.append(LiveBufferSpec(name, pool, _payload(elements, bits), 0, 1,
                                          owner_kind="aic", slots=spec.stages[index],
                                          scale_bytes=scale))
        if b_input_bits != b_bits:
            buffers.append(LiveBufferSpec("b_packed_l1", "l1", _payload(ln * lk, b_input_bits),
                                          0, 1, owner_kind="aic", slots=spec.stages[1]))
        if spec.has_bias:
            buffers.append(LiveBufferSpec("bias", "l1", _payload(tn, acc_bits), 0, 1,
                                          owner_kind="aic", slots=2))
    meta = dict(op.metadata)
    if is_simt:
        buffers.extend(meta.get("simt_buffers", ()))
    if spec.epilogue_kind != "none" or spec.scheduler in {"streamk", "multi_core_splitk"}:
        buffers.extend(meta.get("epilogue_buffers", ()))
        if not meta.get("epilogue_buffers"):
            buffers.append(LiveBufferSpec("epilogue_tile", "ub", _payload(tm * tn, acc_bits),
                                          0, 1, owner_kind="aiv", slots=2))
    work_items = output_blocks * spec.splitk_factor
    candidate = TileCandidate("resident_1pass", TileShape(tm, tn, tk), max(spec.stages),
                              (spec.aic_budget, spec.aiv_budget), work_items, _ceil_div(k, tk),
                              tuple(buffers), logical_flops=2 * m * n * k,
                              uses_simt=is_simt, owner_axis="aiv" if is_simt else "aic",
                              source_guard=spec.template_id)
    return TilePipelineSpec(spec.template_id, (candidate,),
                            simt_ub_carveout_bytes=datapath.simt_ub_carveout_bytes)

def _matrix_shape(op):
    meta = dict(op.metadata)
    shape = tuple(meta.get("matrix_shape", op.local_shape))
    if len(shape) != MATRIX_RANK or min(shape, default=0) <= 0:
        return None
    if op.family not in MATRIX_FAMILIES | GROUPED_FAMILIES and "matrix_shape" not in meta:
        return None
    return shape

def select_catlass_candidates(
    op: OpSpec, datapath: DatapathCapabilitySpec, aic_budget: int, aiv_budget: int,
) -> tuple[CatlassKernelSpec, ...]:
    external = _external_selector("catlass_catalog", "select_catlass_candidates")
    if external is not None:
        return tuple(external(op, datapath, aic_budget, aiv_budget))
    shape = _matrix_shape(op)
    if shape is None or min(aic_budget, aiv_budget) < 0:
        return ()
    meta = dict(op.metadata)
    m, n, k = shape
    dtype_a = meta.get("dtype_a", op.dtype)
    dtype_b = meta.get("dtype_b", op.dtype)
    dtype_acc = meta.get("dtype_acc", "int32" if dtype_a == "int8" else "fp32")
    dtype_c = meta.get("dtype_c", op.dtype)
    arch = datapath.compute_arch.declared
    mode = meta.get("native_matrix_mode", op.dtype)
    cube = bool(aic_budget and arch is not None and arch.supports_native("matrix", mode, dtype_acc))
    modern = datapath.has_l0c_to_ub or datapath.simt_ub_carveout_bytes >= SIMT_UB_CARVEOUT
    if not min(_bits(datapath, dtype_a), _bits(datapath, dtype_b), _bits(datapath, dtype_acc)):
        return ()
    l1 = tuple(meta.get("l1_tile_shape", (MATRIX_BLOCK, MATRIX_BLOCK, L1_K_TILE)))
    l0 = tuple(meta.get("l0_tile_shape", (l1[0], l1[1], MATRIX_BLOCK)))
    if len(l1) != MATRIX_RANK or len(l0) != MATRIX_RANK or min(*l1, *l0) <= 0:
        return ()
    if l1[:2] != l0[:2] or l0[-1] > l1[-1]:
        return ()
    tile_m, tile_n, tile_k = l1
    g_k = _ceil_div(k, tile_k)
    blocks = _ceil_div(m, tile_m) * _ceil_div(n, tile_n)
    acc_bits = _bits(datapath, dtype_acc)
    base = CatlassKernelSpec(
        "catlass.ping_pong", "Ascend950" if modern else "AtlasA2",
        "MmadPingpong" if modern else "MmadAtlasA2Pingpong",
        dtype_a, dtype_b, dtype_c, dtype_acc,
        meta.get("layout_a", "RowMajor"), meta.get("layout_b", "ColumnMajor"),
        meta.get("layout_c", "RowMajor"), meta.get("prologue_a"), meta.get("prologue_b"),
        0, None, 0, l1, l0, (2, 2, 2, 2, 1), bool(meta.get("has_bias")),
        bool(meta.get("enable_unit_flag")), bool(meta.get("l1_resident_slot_match")),
        "dataparallel", 1, 1, "Zn", "normal", "none", (), 0, None,
        aic_budget, aiv_budget, tuple(meta.get("source_refs", ())),
    )
    candidates = []
    if cube and mode not in MX_FORMATS:
        candidates.append(base)
        stages = tuple(meta.get("mutex_stages", (2, 2, 2, 2, 2)))
        if (len(stages) == len(base.stages) and min(stages) >= 1 and
            sum(stages) + 2 * base.has_bias <= MUTEX_LIMIT and
            (not base.enable_unit_flag or stages[-1] == 1)):
            candidates.append(replace(base, template_id="catlass.mutex",
                                      dispatch_policy="MmadPingpongMutex", stages=stages))
        if aiv_budget:
            partials = streamk_crossing_partials(blocks, g_k, aic_budget)
            if partials and g_k >= 2:
                candidates.append(replace(base, template_id="catlass.streamk", scheduler="streamk",
                                          workspace_bytes=_payload(partials * tile_m * tile_n, acc_bits)))
            for split in range(2, min(g_k, aic_budget) + 1):
                candidates.append(replace(base, template_id="catlass.multi_core_splitk",
                                          scheduler="multi_core_splitk", splitk_factor=split,
                                          workspace_bytes=_payload(split * m * n, acc_bits)))
        for direction in ("Zn", "Nz"):
            for offset in meta.get("swizzle_offsets", SWIZZLE_OFFSETS):
                if offset >= 1:
                    candidates.append(replace(base, template_id="catlass.swizzle",
                                              swizzle_direction=direction, swizzle_offset=offset))
    if cube and mode in MX_FORMATS:
        a8w4 = mode == "a8w4_mx"
        base = replace(base, template_id=f"catlass.{mode}", dispatch_policy="MmadA8W4Mx" if a8w4 else "MmadMx",
                       prologue_b="DecompressFp4ToFp8" if a8w4 else base.prologue_b,
                       mx_scale_group=MX_SCALE_GROUP, scale_layout="E8M0_NZ_G32",
                       l1_scale_factor_k=A8W4_SCALE_FACTOR if a8w4 else 1)
        candidates.append(base)
    route = _paired_route(datapath, "l0c", "ub", dtype_acc, aic_budget, aiv_budget)
    evg = tuple(meta.get("evg_tree", ()))
    if cube and aiv_budget and evg and _valid_evg(evg):
        candidates.append(replace(base, template_id="catlass.evg_ub" if route else "catlass.evg_gm",
                                  epilogue_kind="evg_ub" if route else "evg_gm", evg_tree=evg,
                                  workspace_bytes=0 if route else _payload(m * n, acc_bits)))
    groups = tuple(meta.get("group_rows", ()))
    if cube and op.family in GROUPED_FAMILIES and groups and sum(groups) == m and min(groups) >= 0:
        hint = "disable" if all(count <= tile_m for count in groups) else "normal"
        grouped = replace(base, template_id="catlass.grouped_slice_m", scheduler="grouped_slice_m",
                          l2_cache_hint_b=hint)
        candidates.append(grouped)
        if route and mode in MX_FORMATS and meta.get("swiglu_mx_quant"):
            candidates.append(replace(grouped, template_id="catlass.grouped_swiglu_mx_quant",
                                      epilogue_kind="grouped_swiglu_mx_quant"))
    thread_dim = tuple(meta.get("simt_thread_block_dim", ()))
    thread_limit = TLA_SIMT_THREADS if meta.get("simt_backend") == "tla_dsl" else CATLASS_SIMT_THREADS
    if (aiv_budget and datapath.simt_ub_carveout_bytes >= SIMT_UB_CARVEOUT and
        meta.get("simt_buffers") and "simt_workspace_bytes" in meta and
        len(thread_dim) == MATRIX_RANK and min(thread_dim) > 0 and prod(thread_dim) <= thread_limit):
        candidates.append(replace(base, template_id="catlass.simt", dispatch_policy="GatherScatterSIMT",
                                  simt_thread_block_dim=thread_dim, aic_budget=0,
                                  workspace_bytes=int(meta["simt_workspace_bytes"])))
    return tuple(spec for spec in candidates if (
        (pipeline := _matrix_pipeline(spec, op, datapath, shape)) is not None and
        _checked_tiles(pipeline, datapath, spec.aic_budget, spec.aiv_budget, op) is not None
    ))

def _valid_evg(nodes):
    known = set()
    for node in nodes:
        if node.node_id in known or any(parent not in known for parent in node.inputs):
            return False
        known.add(node.node_id)
    return any(node.kind == "acc_load" for node in nodes) and nodes[-1].kind == "aux_store"

def _paired_route(datapath, src, dst, dtype, aic, aiv, tile=None):
    try:
        if (src, dst) in {("l0c", "ub"), ("ub", "l1")}:
            owner = next(({"aic": aic_id, "aiv": aiv_id}
                          for aic_id, aiv_ids in datapath.aic_aiv_pairs if aic_id < aic
                          for aiv_id in aiv_ids if aiv_id < aiv), None)
            return owner is not None and bool(datapath.route(src, dst, dtype, owner_group=owner,
                                                              tile=tile))
        owner_count = aiv if src == "ub" else aic
        return owner_count > 0 and bool(datapath.route(src, dst, dtype, owner_group=0, tile=tile))
    except ModelingError:
        return False

def _attn_res_tiles(op, datapath, aiv):
    if (not op.local_shape or op.local_shape[-1] != ATTN_RES_WIDTH
        or op.dtype not in {"bf16", "bfloat16"} or op.layout != "contiguous"):
        return None
    width = ATTN_RES_WIDTH
    rows = prod(op.local_shape[:-1])
    fp_cols = _ceil_div(width, ATTN_RES_STAT_GRANULE) * ATTN_RES_STAT_GRANULE
    bf_cols = _ceil_div(width, BF16_COL_GRANULE) * BF16_COL_GRANULE
    fixed = FP32_ELEMENT_BYTES * fp_cols + 2 * bf_cols
    room = datapath.usable_bytes("ub")
    per_row = 2 * FP32_ELEMENT_BYTES * fp_cols + 2 * bf_cols
    max_rows = min(rows, ATTN_RES_MAX_TILE_ROWS, max(0, (room - fixed) // per_row))
    for block in range(max_rows, 0, -1):
        tiles = _ceil_div(rows, block)
        depth = 1 if tiles == 1 else 2
        stat = _ceil_div(block, ATTN_RES_STAT_GRANULE) * ATTN_RES_STAT_GRANULE
        stage = block * per_row + ATTN_RES_STAT_BYTES * stat
        if fixed + depth * stage <= room:
            buffers = (LiveBufferSpec("fixed", "ub", fixed, 0, 1, owner_kind="aiv"),
                       LiveBufferSpec("row_pipeline", "ub", stage, 0, 1, owner_kind="aiv", slots=depth))
            return TilePipelineSpec("block_attn_res_update_rms_norm", (
                TileCandidate("resident_1pass", TileShape(block, width, 1), depth,
                              (0, aiv), tiles, 1, buffers, owner_axis="aiv"),
            ))
    return None

def _dsl_spec(kernel, meta, op, datapath, aic_budget, aiv_budget):
    mode = DSL_KERNELS[kernel]
    if not datapath.has_l0c_to_ub or not aiv_budget or (mode == "mix_cube_vector" and not aic_budget):
        return None
    pipeline = meta.get("dsl_tile_spec")
    workspace = meta.get("dsl_workspace_bytes")
    if pipeline is None and kernel == "block_attn_res_update_rms_norm" and op.family == kernel:
        if "ub" in dict(datapath.buffers):
            pipeline, workspace = _attn_res_tiles(op, datapath, aiv_budget), 0
    if pipeline is None or workspace is None or workspace < 0:
        return None
    aic = 0 if mode == "vector_only" else aic_budget
    tile = _checked_tiles(pipeline, datapath, aic, aiv_budget, op)
    if tile is None or not tile.pool_peaks:
        return None
    candidate = next((candidate for candidate in pipeline.candidates
                      if candidate.algorithm == tile.algorithm and candidate.tile_shape == tile.tile_shape
                      and candidate.depth == tile.depth), None)
    if candidate is None:
        return None
    shape = tile.tile_shape
    pools = {peak.pool: peak.peak_bytes for peak in tile.pool_peaks}
    routes = tuple(meta.get("dsl_direct_routes", ()))
    dims = (shape.m, shape.n, shape.k)
    if any(not _paired_route(datapath, src, dst, op.dtype, aic, aiv_budget, dims)
           for _, src, dst in routes):
        return None
    return DslKernelSpec(
        f"dsl.{kernel}", mode, tile.algorithm, meta.get("dsl_variant", "default"), dims,
        tuple((buf.name, buf.pool, buf.slots) for buf in candidate.buffers), routes,
        tuple(meta.get("dsl_l2_cache_policies", ())), int(workspace),
        bool(meta.get("uses_aicpu_metadata")), bool(meta.get("has_eager_d2h_readback")),
        candidate.uses_simt, aic, aiv_budget, pools.get("ub", 0), pools.get("l1", 0),
        pools.get("l0c", 0), tuple(meta.get("source_refs", ())),
    )

def _convex_members(ops, ids):
    if len(ids) != len(set(ids)) or not ids:
        return ()
    members = set(ids)
    by_id = {op.op_id: op for op in ops}
    ordered = tuple(op for op in ops if op.op_id in members)
    if tuple(op.op_id for op in ordered) != ids or any(
        op.link_bytes or dict(op.metadata).get("collective_barrier") for op in ordered
    ):
        return ()
    internal = set()
    outside = []
    for op in ordered:
        if any(dep in members and dep not in internal for dep in op.deps):
            return ()
        internal.add(op.op_id)
        outside.extend(dep for dep in op.deps if dep not in members)
    seen = set()
    while outside:
        dep = outside.pop()
        if dep in members:
            return ()
        if dep in by_id and dep not in seen:
            seen.add(dep)
            outside.extend(by_id[dep].deps)
    return ordered

def select_dsl_candidates(
    ops: Sequence[OpSpec], datapath: DatapathCapabilitySpec, aic_budget: int, aiv_budget: int,
) -> tuple[tuple[DslKernelSpec, tuple[str, ...]], ...]:
    external = _external_selector("dsl_catalog", "select_dsl_candidates")
    if external is not None:
        return tuple(external(ops, datapath, aic_budget, aiv_budget))
    candidates = []
    for op in ops:
        meta = dict(op.metadata)
        for variant in meta.get("dsl_variants", ({},)):
            effective = {**meta, **variant}
            kernel = effective.get("dsl_kernel", op.family if op.family in DSL_KERNELS else op.recipe)
            members = tuple(effective.get("dsl_members", (op.op_id,)))
            if kernel in DSL_KERNELS and _convex_members(ops, members):
                spec = _dsl_spec(kernel, effective, op, datapath, aic_budget, aiv_budget)
                if spec is not None:
                    candidates.append((spec, members))
    return tuple(candidates)

def _general_fuse(ops, datapath, aic, aiv):
    for op in ops:
        for declaration in dict(op.metadata).get("fuse_regions", ()):
            ids = tuple(declaration["op_ids"])
            members = _convex_members(ops, ids)
            routes = tuple(declaration["edge_routes"])
            if len(members) < 2 or not declaration.get("numerically_equivalent"):
                continue
            if len({item.stage for item in members}) != 1 or not routes:
                continue
            edges = {(dep, item.op_id) for item in members for dep in item.deps if dep in ids}
            if {(route.src_op, route.dst_op) for route in routes} != edges:
                continue
            # The standalone compiler accepts forward immutable handoffs.
            # Mutation aliases require the interpreter's full port analysis.
            if any(route.in_place_mutates or route.ring_depth < 1 for route in routes):
                continue
            tile = _checked_tiles(declaration["tile_spec"], datapath, aic, aiv, op)
            if tile is None or not tile.pool_peaks:
                continue
            shape = tile.tile_shape
            if any(route.route_kind != "gm_spill" and not _paired_route(
                datapath, *route.route_kind.split("->"), route.representation, aic, aiv,
                (shape.m, shape.n, shape.k),
            ) for route in routes):
                continue
            spec = FuseRegionSpec(declaration["region_id"], ids,
                                  tuple(declaration.get("live_in_obligations", ())),
                                  tuple(declaration.get("live_out_obligations", ())),
                                  routes, tile.algorithm, tile, declaration["backend"])
            yield spec, tuple(declaration.get("preserve_ports", ()))

def _dsl_proposals(ops, datapath, aic, aiv):
    from mindie_sim.rewrite.schema import FuseRegion, KernelBoundary, OpRef, SelectTemplate

    for spec, ids in select_dsl_candidates(ops, datapath, aic, aiv):
        refs = tuple(OpRef("atom", op_id) for op_id in ids)
        meta = next(dict(op.metadata) for op in ops if op.op_id == ids[-1])
        if len(refs) == 1:
            yield (SelectTemplate(refs[0], "K3", spec.kernel_id, (
                ("variant", spec.template_variant), ("aic_budget", spec.aic_budget),
                ("aiv_budget", spec.aiv_budget),
            )),), spec
        elif "dsl_preserve_ports" in meta:
            preserve = tuple(meta["dsl_preserve_ports"])
            yield (FuseRegion(refs, preserve, (KernelBoundary(spec.kernel_id, refs),),
                              fusion_tier="K3_dsl_pipeline"),), spec
    for spec, preserve in _general_fuse(ops, datapath, aic, aiv):
        members = tuple(OpRef("atom", op_id) for op_id in spec.op_ids)
        yield (FuseRegion(members, preserve, (KernelBoundary(spec.region_id, members),),
                          fusion_tier="K3_dsl_pipeline"),), spec

def _catlass_proposals(ops, datapath, aic, aiv, *, registered=None):
    from mindie_sim.rewrite.schema import OpRef, PipelineTiles, SelectTemplate

    for op in ops:
        candidates = select_catlass_candidates(op, datapath, aic, aiv)
        first_by_template = {spec.template_id: spec for spec in reversed(candidates)}
        ordered = sorted(candidates, key=lambda spec: spec != first_by_template[spec.template_id])
        for spec in ordered:
            if registered is not None and spec.template_id not in registered:
                continue
            ref = OpRef("atom", op.op_id)
            params = (
                ("scheduler", spec.scheduler), ("splitk_factor", spec.splitk_factor),
                ("swizzle_offset", spec.swizzle_offset), ("swizzle_direction", spec.swizzle_direction),
                ("aic_budget", spec.aic_budget), ("aiv_budget", spec.aiv_budget),
            )
            pipeline = _matrix_pipeline(spec, op, datapath, _matrix_shape(op))
            if pipeline is not None:
                yield (SelectTemplate(ref, "K2", spec.template_id, params), PipelineTiles(ref, pipeline)), spec

def solve_dsl(
    dossier: ScenarioDossier, fw_result: FrameworkSolveResult, *, dual_report=None,
    dual_program=None, event_dag=None, dual_certificate=None, max_trials: int = DEFAULT_MAX_TRIALS,
) -> DslSolveResult:
    from mindie_sim.rewrite.schema import plan_sha256

    search = _TrialSearch(dossier, max_trials, dual_program=dual_program, event_dag=event_dag,
                          dual_certificate=dual_certificate)
    search.seed(fw_result.best_plan, fw_result.best_eval)
    scenario = _resolved(dossier)
    ops = _atomic_ops(scenario)
    paths = tuple(datapath_for_die(device) for device in getattr(_machine(scenario), "dies", ()))
    specs_by_edit = {}
    for index, stage in enumerate(DSL_STAGES):
        base = search.best_plan
        limit = search.remaining // (len(DSL_STAGES) - index)
        if index == len(DSL_STAGES) - 1:
            applied = getattr(search.best_eval, "applied_plan", getattr(search.best_eval, "applied", None))
            placed_ops = _unique_ops(applied.executable) if applied is not None else ops
            atomic_ids = {op.op_id for op in ops}
            handles = ()
            if applied is not None:
                from mindie_sim.rewrite.schema import OpRef
                handles = tuple((op_id, OpRef("result_of", edit_id))
                                for edit_id, output in applied.handles
                                for op_id in output if len(output) == 1 and op_id not in atomic_ids)
            search.stage(stage, (_stream_plan(base, stage, operation)
                                 for operation in _stream_candidates(scenario, placed_ops, handles)), limit)
        elif paths:
            path = paths[0]
            aic, aiv = min(p.aic_count for p in paths), min(p.aiv_count for p in paths)
            if index == 0:
                proposals = _dsl_proposals(ops, path, aic, aiv)
            else:
                book = getattr(search.handle, "recipe_book", None)
                bottlenecks = dict(getattr(dossier, "bottleneck_ops", ()))
                sites = sorted(ops, key=lambda op: -bottlenecks.get(op.op_id, 0))
                proposals = _catlass_proposals(sites, path, aic, aiv,
                                               registered=getattr(book, "recipes", None))

            def plans():
                for operations, spec in proposals:
                    candidate = _with_operations(base, stage, operations)
                    specs_by_edit[plan_sha256(candidate)] = spec
                    yield candidate

            search.stage(stage, plans(), limit)
        else:
            search.stage(stage, (), limit)
    specs = tuple(specs_by_edit[digest] for _, _, digest in search.history if digest in specs_by_edit)
    specs = tuple(dict.fromkeys(specs))
    return DslSolveResult(search.best_plan, search.best_eval, _wall(search.best_eval),
                          tuple(s for s in specs if isinstance(s, CatlassKernelSpec)),
                          tuple(s for s in specs if isinstance(s, DslKernelSpec)),
                          tuple(s for s in specs if isinstance(s, FuseRegionSpec)),
                          tuple(search.history), search.tried)
