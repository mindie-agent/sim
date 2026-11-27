"""Shared-expert W4A8 single-WeightNz matrix leaf.

[Role] Bind the ordinary shared-expert ``npu_quant_matmul`` W4A8 source call
  (FP8 E4M3 activation x FP4 E2M1 FRACTAL_NZ weight, two E8M0 group-32 scales,
  BF16 output, no expert group list) to the existing finite W4A8 matrix/B8
  physical components.
[Invariants] Only shared-expert gate_up/down ordinary projections with the
  exact source ABI select this plan.  There is no routed-expert row list, no
  SiTU/quant fusion and no grouped controller.  Output tiles and participating
  AIC owners follow the QuantBatchMatmulV4 MX SWAT host tiler and compact
  tail-split scheduler; the physical matrix/B8 panel component is shared with
  the A5 ABB5 W4A8 leaf.
[Timeouts & Locks] Bounded immutable panel/ledger construction; no waits.
[Side-Effect Policy] Returns immutable Work/Program metadata; no device I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef

from mindie_sim.prediction.instructions import InstructionInventory
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.operators.matrix.refill import RefillPipeline
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from .mxfp4_geometry import (
    CUBE_MN_FRACTAL, MX_SCALE_GROUP, MX_SCALE_GROUP_PAIR, Config, align, ceildiv,
)
from .mxfp4_grouped import B8_NATIVE_I8_PER_CYCLE_PER_VF, B8_PROLOGUE_DUPLICATE, _US_PER_SECOND
from .mxfp4_registers import B8_WIDEN_BLOCK_ELEMENTS, CLASSES, TABLES, widen

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py",
        source_sha256="3d23f23d108ac0718ad85fe0c25b2fc9ebc7699cd640bc74e9967819192bc19b",
    ),
    SourceRef(
        repository="cann/ops-nn",
        revision="238f76a06d0cfc5ee90071e2c08310e2b517c3a6",
        relative_path="matmul/quant_batch_matmul_v4/op_host/op_tiling/arch35/quant_batch_matmul_v4_weight_quant_mx_swat_tiling_solver.cpp",
        source_sha256="6807a0acef29352df1602536c82ef07ab5768a1ab85fffe5a3f4a317a67f7224",
    ),
    SourceRef(
        repository="cann/ops-tensor",
        revision="c7798ac7b1b53a73a4a69eac2ca0b405393e09b0",
        relative_path="include/blaze/gemm/block/block_scheduler_matmul_swat_with_tail_split.h",
        source_sha256="df7d855fad4514f125bda11f7ea284c8343fa0a98c8d071290e82bda1d5faa9d",
    ),
)

MX_SCALE_PAIR_ROUNDUP = MX_SCALE_GROUP_PAIR - 1
OUTPUT_N_ALIGNMENT = 64
SWAT_BASIC_BLOCK_128 = 128
SWAT_BASIC_BLOCK_256 = 256
SWAT_K_ALIGN_MX = 8
SWAT_BASEK_LIMIT = 4095
SWAT_STEPK_THRESHOLD = 4
SWAT_SCALE_FACTOR_MAX = 4
SWAT_TAIL_WINDOW_LEN = 4
SWAT_MTE2_CACHELINE_BYTES = 128
SWAT_L1_HALF_BYTES = 256 * 1024
SWAT_L1_DOUBLE_BUFFER = 2
SWAT_L1_FOUR_BUFFER = 4
SWAT_FP32_BYTES = 4

KIND = "ascend_shared_w4a8_mxfp_single_nz"
SWAT_L0_DOUBLE_BUFFER = 2
SWAT_WIDEN_PROLOGUE_MASK = 2
SWAT_SCALE_PAIR_COLS = 2
SOURCE = {
    "caller": {
        "repository": "vllm-ascend-v027",
        "revision": "v027/68a158be",
        "path": "vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py",
        "sha256": "68a158be11cbf33bace3032b5b89d32f5d21a0ab206b9777cebbd5520ad0f374",
        "lines": "41-105",
        "call": "torch_npu.npu_quant_matmul(x FP8, weight FP4 NZ, "
                "weight_scale E8M0, pertoken_scale E8M0, "
                "x2_dtype=float4_e2m1fn_x2, group_sizes=[0,0,32], "
                "output_dtype=BF16)",
    },
    "weight_transform": {
        "call": "npu_format_cast(...,29, customize_dtype=fp8_e4m3fn, "
                "input_dtype=float4_e2m1fn_x2); transpose(-1,-2); "
                "weight_scale.reshape(n,k//2,2).transpose(-3,-2)",
        "storage": "FRACTAL_NZ FP4 E2M1, block32 weight scale E8M0",
    },
    "physical_component": {
        "source": "cann/ops-tensor KernelMixWithWeightPrologue + BlockMmadWeightPrologueMx + ShiftW4ToW8",
        "status": "swat_weight_prologue_mx_aic_scale_b_and_k_l1_base_k",
    },
    "host_binary": {
        "repository": "cann/ops-nn",
        "revision": "238f76a06d0cfc5ee90071e2c08310e2b517c3a6",
        "path": "matmul/quant_batch_matmul_v4/op_host/op_tiling/arch35/"
                "quant_batch_matmul_v4_weight_quant_mx_swat_tiling_solver.cpp",
        "status": "swat_host_tiler_and_tail_split_scheduler",
    },
}

SOURCE_INTERFACE = {
    "activation_format": "fp8_e4m3",
    "input_format": "fp8_e4m3",
    "activation_group_size": 32,
    "activation_scale_format": "fp8_e8m0",
    "weight_format": "mxfp4",
    "weight_dtype": "mxfp4",
    "weight_scale_format": "fp8_e8m0",
    "weight_scale_storage_format": "fp8_e8m0",
    "weight_representation": "fp4_group32_packed",
    "compute_format": "fp8_e4m3",
    "output_format": "bf16",
}
_NATIVE_COMPUTE_DTYPES = frozenset({"", "fp8_e4m3", "fp8_e4m3fn"})
SHARED_FAMILIES = ("gate_up_projection", "down_projection")


def _matrix_shape(op):
    if len(op.local_shape) != 3:
        return None
    m, n, k = (int(value) for value in op.local_shape)
    return (m, n, k) if m > 0 and n > 0 and k > 0 else None


def _source_declaration_valid(meta, m, n, k):
    """Validate the typed declaration once, including explicit bad flags."""
    k_tail = (k + MX_SCALE_PAIR_ROUNDUP) // MX_SCALE_GROUP_PAIR
    expected_xscale = (m, k_tail, 2)
    shape = meta.get("activation_scale_shape")
    if shape is None or tuple(shape) != expected_xscale:
        return False
    wshape = meta.get("weight_scale_shape")
    if wshape is not None and tuple(wshape) != (n, k_tail, 2):
        return False
    layout = meta.get("weight_layout")
    if layout not in (None, "FRACTAL_NZ", "NZ"):
        return False
    if meta.get("bias") not in (None, False, 0):
        return False
    if meta.get("bias_shape") not in (None, (), []):
        return False
    return True


def matches_source_interface(op):
    """Exact ordinary shared-expert W4A8 source interface.

    The shared-expert caller has no matrix_group_rows: it is one ordinary
    WeightNz matrix, not the routed GroupedMatmul branch.  W8, INT8, group-64
    weights, non-NZ explicit layouts, bias and malformed E8M0 scale shapes do
    not match.
    """
    if str(getattr(op, "family", "")) not in SHARED_FAMILIES:
        return False
    shape = _matrix_shape(op)
    if shape is None:
        return False
    m, n, k = shape
    meta = dict(op.metadata)
    if meta.get("ffn_category") != "shared_expert":
        return False
    if meta.get("quant_category") != "shared_expert":
        return False
    if meta.get("matrix_group_rows") or meta.get("matrix_groups", 1) != 1:
        return False
    if meta.get("native_fusion_group") or meta.get("native_fusion"):
        return False
    if any(meta.get(key) != value for key, value in SOURCE_INTERFACE.items()):
        return False
    abi = getattr(op, "precision_abi", None)
    if abi is not None and str(getattr(abi, "compute_dtype", "") or "").lower() not in _NATIVE_COMPUTE_DTYPES:
        return False
    if meta.get("weight_block") not in (None, MX_SCALE_GROUP):
        return False
    if meta.get("weight_group_size") not in (None, MX_SCALE_GROUP):
        return False
    if not _source_declaration_valid(meta, m, n, k):
        return False
    if k % MX_SCALE_GROUP_PAIR or n % OUTPUT_N_ALIGNMENT:
        return False
    return True


def config_from_op(op):
    if not matches_source_interface(op):
        raise ModelingError("shared_w4a8_requires_fp8_mxfp4_group32_e8m0_bf16")
    m, n, k = _matrix_shape(op)
    if k % MX_SCALE_GROUP_PAIR or n % OUTPUT_N_ALIGNMENT:
        raise ModelingError("shared_w4a8_requires_k64_n64")
    return Config(k, n, (m,)), (m, n, k)


@dataclass(frozen=True)
class _SwatTiling:
    base_m: int
    base_n: int
    base_k: int
    k_l1: int
    scale_k_l1: int
    n_bub_size: int
    k_bub_size: int
    m_blocks: int
    n_blocks: int
    used_cores: int
    m_tail_tile: int
    n_tail_tile: int
    m_base_tail_split: int
    n_base_tail_split: int
    m_tail_main: int
    n_tail_main: int
    l1_buffer_num: int
    ub_buffer_num: int
    disable_weight_l2: bool


def _floor_align(value: int, granule: int) -> int:
    return (value // granule) * granule


def _swat_logical_coord(tile_idx: int, m_cnt: int, n_cnt: int) -> tuple[int, int]:
    m_core = min(SWAT_TAIL_WINDOW_LEN, m_cnt)
    main_row = m_cnt // m_core - 1
    m_tail_core = m_cnt - m_core * main_row
    row_idx = tile_idx // (m_core * n_cnt)
    if row_idx < main_row:
        local_idx = tile_idx - row_idx * m_core * n_cnt
        m_idx = row_idx * m_core + local_idx % m_core
        n_idx = (local_idx // m_core) % n_cnt
    else:
        row_idx = main_row
        tail_idx = tile_idx - main_row * m_core * n_cnt
        m_idx = main_row * m_core + tail_idx % m_tail_core
        n_idx = (tail_idx // m_tail_core) % n_cnt
    if row_idx & 1:
        n_idx = n_cnt - 1 - n_idx
    return m_idx, n_idx


def _swat_axis_extent(
    idx: int, total_size: int, base_size: int, block_cnt: int, tail_split: int, tail_main: int,
) -> int:
    norm_cnt = block_cnt - tail_split
    if idx < norm_cnt:
        return base_size
    merge_size = total_size - norm_cnt * base_size
    main_sz = merge_size if tail_split == 1 else tail_main
    return main_sz if idx < block_cnt - 1 else merge_size - (tail_split - 1) * main_sz


def _swat_axis_origin(idx: int, base_size: int, norm_cnt: int, tail_main: int) -> int:
    if idx > norm_cnt:
        return idx * base_size - (idx - norm_cnt) * (base_size - tail_main)
    return idx * base_size


def _swat_valid_split(single_m: int, single_n: int, m_tile: int, n_tile: int) -> tuple[int, int]:
    m_split = align(ceildiv(single_m, m_tile), CUBE_MN_FRACTAL)
    n_split = align(ceildiv(single_n, n_tile), CUBE_MN_FRACTAL)
    return min(m_tile, ceildiv(single_m, m_split)), min(n_tile, ceildiv(single_n, n_split))


def _swat_is_l0_feasible(base_m: int, base_n: int, base_k: int, buffers: dict[str, int]) -> bool:
    return (
        base_m * base_k * 2 <= buffers["l0a"]
        and base_n * base_k * 2 <= buffers["l0b"]
        and base_m * base_n * SWAT_FP32_BYTES <= buffers["l0c"]
    )


def _swat_is_l1_feasible(
    base_m: int, base_n: int, k_l1: int, scale_k_l1: int, l1_bufs: int, l1_size: int,
) -> bool:
    if l1_size < SWAT_L1_HALF_BYTES:
        return False
    k_aligned = align(k_l1, MX_SCALE_GROUP_PAIR)
    scale_k_aligned = align(scale_k_l1, MX_SCALE_GROUP_PAIR)
    scale_pair = (base_m + base_n) * scale_k_aligned // MX_SCALE_GROUP
    half_use = (l1_bufs // 2) * (base_m + base_n) * k_aligned + scale_pair
    half_limit = min(SWAT_L1_HALF_BYTES, l1_size - SWAT_L1_HALF_BYTES)
    return half_use <= half_limit


def _swat_find_k_bub_nz(n_bl1: int, k_bl1: int, ub_bufs: int, ub_size: int) -> int:
    def valid(n_sz: int, k_sz: int) -> bool:
        if n_sz <= 0 or k_sz <= 0 or k_sz % SWAT_K_ALIGN_MX != 0:
            return False
        n_al = align(n_sz, CUBE_MN_FRACTAL)
        k_blk = align(k_sz, MX_SCALE_GROUP)
        k_l1 = align(k_sz, MX_SCALE_GROUP_PAIR)
        return ub_bufs * (n_al * k_blk // 2 + n_al * k_l1) <= ub_size

    if k_bl1 <= MX_SCALE_GROUP_PAIR:
        return k_bl1 if valid(n_bl1, k_bl1) else 0
    min_k = align(ceildiv(k_bl1, 2), MX_SCALE_GROUP_PAIR)
    if min_k >= k_bl1 or min_k % SWAT_K_ALIGN_MX != 0:
        return 0
    return min_k if valid(n_bl1, min_k) else 0


def _swat_calc_l1(
    n: int, k: int, base_m: int, base_n: int, base_k: int,
    l1_bufs: int, ub_bufs: int, buffers: dict[str, int],
) -> tuple[int, int, int, int] | None:
    max_step_k = min(SWAT_STEPK_THRESHOLD, ceildiv(k, base_k))
    n_bl1 = min(n, base_n)
    for step_k in range(max_step_k, 0, -1):
        k_l1 = step_k * base_k
        k_bub = _swat_find_k_bub_nz(n_bl1, min(k, k_l1), ub_bufs, buffers["ub"])
        if not k_bub:
            continue
        max_scale = max(1, min(SWAT_SCALE_FACTOR_MAX, ceildiv(k, k_l1)))
        for scale_factor in range(max_scale, 0, -1):
            scale_k_l1 = k_l1 * scale_factor
            if _swat_is_l1_feasible(base_m, base_n, k_l1, scale_k_l1, l1_bufs, buffers["l1"]):
                return k_l1, scale_k_l1, n_bl1, k_bub
    return None


def _swat_adjust_basic_block(
    m: int, n: int, k: int, base_m: int, base_n: int, base_k: int,
    cores: int, buffers: dict[str, int],
) -> tuple[int, int, int]:
    m_max_tile = ceildiv(m, CUBE_MN_FRACTAL)
    n_max_tile = ceildiv(n, CUBE_MN_FRACTAL)
    m_cnt = max(1, ceildiv(m, base_m))
    n_cnt = max(1, ceildiv(n, base_n))
    if m_max_tile > n_max_tile:
        temp_n = align(ceildiv(n, n_cnt), CUBE_MN_FRACTAL)
        n_cnt = max(1, ceildiv(n, temp_n))
        m_cnt = max(1, cores // n_cnt)
        temp_m = align(ceildiv(m, m_cnt), CUBE_MN_FRACTAL)
    else:
        temp_m = align(ceildiv(m, m_cnt), CUBE_MN_FRACTAL)
        m_cnt = max(1, ceildiv(m, temp_m))
        n_cnt = max(1, cores // m_cnt)
        temp_n = align(ceildiv(n, n_cnt), CUBE_MN_FRACTAL)
    m_cnt = max(1, ceildiv(m, temp_m))
    n_cnt = max(1, ceildiv(n, temp_n))
    while temp_n > temp_m * 2 and n_cnt < cores // 2 and temp_n != CUBE_MN_FRACTAL:
        n_cnt *= 2
        m_cnt = max(1, cores // n_cnt)
        temp_m = align(ceildiv(m, m_cnt), CUBE_MN_FRACTAL)
        temp_n = align(ceildiv(n, n_cnt), CUBE_MN_FRACTAL)
        m_cnt, n_cnt = max(1, ceildiv(m, temp_m)), max(1, ceildiv(n, temp_n))
    while temp_m >= temp_n * 2 and m_cnt < cores // 2 and temp_m != CUBE_MN_FRACTAL:
        m_cnt *= 2
        n_cnt = max(1, cores // m_cnt)
        temp_m = align(ceildiv(m, m_cnt), CUBE_MN_FRACTAL)
        temp_n = align(ceildiv(n, n_cnt), CUBE_MN_FRACTAL)
        m_cnt, n_cnt = max(1, ceildiv(m, temp_m)), max(1, ceildiv(n, temp_n))
    k_align = align(k, SWAT_BASIC_BLOCK_128)
    k_max = _floor_align((buffers["l0a"] // 2) // max(temp_m, temp_n), SWAT_BASIC_BLOCK_128)
    cand_k = min(k_align, k_max)
    if k_max >= SWAT_BASIC_BLOCK_128 and _swat_is_l0_feasible(temp_m, temp_n, cand_k, buffers):
        if cand_k > SWAT_BASEK_LIMIT:
            cand_k = align(cand_k // 2, SWAT_BASIC_BLOCK_256)
        return temp_m, temp_n, cand_k
    return base_m, base_n, base_k


def _swat_optimize_edge(
    total_size: int, base_size: int, block_cnt: int, is_inner_aligned: bool,
) -> tuple[int, int]:
    tail_size = total_size % base_size
    if block_cnt <= 1 or tail_size == 0 or not is_inner_aligned:
        return 1, 0
    max_cnt = min((base_size - tail_size) // CUBE_MN_FRACTAL, block_cnt)
    window = min(SWAT_TAIL_WINDOW_LEN, block_cnt)
    main_win = block_cnt // window - 1
    tail_win = block_cnt - main_win * window
    best_perf = (main_win + 1) * base_size
    tail_main, split_cnt = 0, 1
    merge_win, merge_len = 1, tail_win - 1
    while merge_len < max_cnt:
        cand_main = align(ceildiv(merge_len * base_size + tail_size, merge_len + 1), CUBE_MN_FRACTAL)
        cur_perf = (main_win + 1 - merge_win) * base_size + merge_win * cand_main
        if cur_perf <= best_perf:
            best_perf, tail_main, split_cnt = cur_perf, cand_main, merge_len + 1
        merge_len += window
        merge_win += 1
    return split_cnt, tail_main


def _swat_calc_tail_tile(
    m: int, n: int, base_m: int, base_n: int, m_cnt: int, n_cnt: int,
    m_split: int, m_tail_main: int, n_split: int, n_tail_main: int, cores: int,
) -> tuple[int, int, int]:
    total_blocks = m_cnt * n_cnt
    tail_blocks = total_blocks % cores
    base_round = total_blocks - tail_blocks

    def used_for(m_tile: int, n_tile: int) -> int:
        used = 0
        for tail_idx in range(tail_blocks):
            mi, ni = _swat_logical_coord(base_round + tail_idx, m_cnt, n_cnt)
            sm = _swat_axis_extent(mi, m, base_m, m_cnt, m_split, m_tail_main)
            sn = _swat_axis_extent(ni, n, base_n, n_cnt, n_split, n_tail_main)
            vm, vn = _swat_valid_split(sm, sn, m_tile, n_tile)
            used += vm * vn
        return used

    if tail_blocks == 0:
        return 1, 1, cores
    m_tail_sz = m - (m_cnt - 1) * base_m
    n_tail_sz = n - (n_cnt - 1) * base_n
    split_m_first = m_tail_sz >= n_tail_sz
    m_tile_max, n_tile_max = ceildiv(base_m, CUBE_MN_FRACTAL), ceildiv(base_n, CUBE_MN_FRACTAL)
    pre_max, sec_max = (m_tile_max, n_tile_max) if split_m_first else (n_tile_max, m_tile_max)
    pre, sec = 1, 1
    updated = True
    while updated:
        updated = False
        cur_m, cur_n = (pre, sec) if split_m_first else (sec, pre)
        cur_used = used_for(cur_m, cur_n)
        cand_m, cand_n = (pre + 1, sec) if split_m_first else (sec, pre + 1)
        cand_used = used_for(cand_m, cand_n)
        if pre < pre_max and cur_used < cand_used <= cores:
            pre, cur_used, updated = pre + 1, cand_used, True
        cand_m, cand_n = (pre, sec + 1) if split_m_first else (sec + 1, pre)
        cand_used = used_for(cand_m, cand_n)
        if sec < sec_max and cur_used < cand_used <= cores:
            sec, updated = sec + 1, True
    m_tile, n_tile = (pre, sec) if split_m_first else (sec, pre)
    used_cores = cores if total_blocks >= cores else used_for(m_tile, n_tile)
    return m_tile, n_tile, used_cores


def _swat_try_solve(
    m: int, n: int, k: int, cores: int, buffers: dict[str, int],
    l1_bufs: int, ub_bufs: int, *, conservative: bool,
) -> _SwatTiling | None:
    if not conservative:
        base_m = align(min(m, SWAT_BASIC_BLOCK_256), CUBE_MN_FRACTAL)
        base_n = align(min(n, SWAT_BASIC_BLOCK_256), CUBE_MN_FRACTAL)
        base_k = align(min(k, SWAT_BASIC_BLOCK_128), SWAT_K_ALIGN_MX)
        if ceildiv(m, base_m) * ceildiv(n, base_n) < cores:
            base_m, base_n, base_k = _swat_adjust_basic_block(
                m, n, k, base_m, base_n, base_k, cores, buffers)
    else:
        base_m, base_n = CUBE_MN_FRACTAL, CUBE_MN_FRACTAL
        base_k = min(k, MX_SCALE_GROUP_PAIR)
    if not _swat_is_l0_feasible(base_m, base_n, base_k, buffers):
        return None
    m_cnt, n_cnt = ceildiv(m, base_m), ceildiv(n, base_n)
    if conservative or (m_cnt == 1 and n_cnt == 1):
        m_split, m_tail_main, n_split, n_tail_main = 1, 0, 1, 0
    else:
        inner_aligned = (k % SWAT_MTE2_CACHELINE_BYTES) == 0
        m_split, m_tail_main = _swat_optimize_edge(m, base_m, m_cnt, inner_aligned)
        n_split, n_tail_main = _swat_optimize_edge(n, base_n, n_cnt, inner_aligned)
    m_tile, n_tile, used_cores = _swat_calc_tail_tile(
        m, n, base_m, base_n, m_cnt, n_cnt,
        m_split, m_tail_main, n_split, n_tail_main, cores)
    l1_quad = _swat_calc_l1(n, k, base_m, base_n, base_k, l1_bufs, ub_bufs, buffers)
    if l1_quad is None:
        return None
    k_l1, scale_k_l1, n_bub, k_bub = l1_quad
    return _SwatTiling(
        base_m=base_m, base_n=base_n, base_k=base_k, k_l1=k_l1, scale_k_l1=scale_k_l1,
        n_bub_size=n_bub, k_bub_size=k_bub,
        m_blocks=m_cnt, n_blocks=n_cnt, used_cores=used_cores,
        m_tail_tile=m_tile, n_tail_tile=n_tile,
        m_base_tail_split=m_split, n_base_tail_split=n_split,
        m_tail_main=m_tail_main, n_tail_main=n_tail_main,
        l1_buffer_num=l1_bufs, ub_buffer_num=ub_bufs,
        disable_weight_l2=(base_m >= m),
    )


def _solve_swat_tiling(m: int, n: int, k: int, cores: int, buffers: dict[str, int]) -> _SwatTiling:
    """QuantBatchMatmulV4WeightQuantMxSwatTiling::DoOpTiling (2-buf then 4-buf fallback)."""
    for l1_bufs, ub_bufs, conservative in (
        (SWAT_L1_DOUBLE_BUFFER, SWAT_L1_DOUBLE_BUFFER, False),
        (SWAT_L1_FOUR_BUFFER, SWAT_L1_FOUR_BUFFER, False),
        (SWAT_L1_FOUR_BUFFER, SWAT_L1_FOUR_BUFFER, True),
    ):
        solved = _swat_try_solve(m, n, k, cores, buffers, l1_bufs, ub_bufs, conservative=conservative)
        if solved is not None:
            return solved
    raise ModelingError("shared_w4a8_swat_tiling_infeasible")


def _swat_owner_schedule(
    m: int, n: int, tiling: _SwatTiling, cores: int,
) -> tuple[tuple[tuple[tuple[int, int, int], ...], ...], tuple[int, ...], tuple[tuple[int, int], ...]]:
    """BlockSchedulerMatmulSwatWithTailSplit + RunAic per-owner tile schedule."""
    block_num = tiling.used_cores
    total_cnt = tiling.m_blocks * tiling.n_blocks
    tail_cnt = total_cnt % block_num
    base_round = total_cnt - tail_cnt
    m_split = min(max(1, tiling.m_base_tail_split), tiling.m_blocks)
    n_split = min(max(1, tiling.n_base_tail_split), tiling.n_blocks)
    n_norm = tiling.n_blocks - n_split

    def block_dims(logical_idx: int) -> tuple[int, int, int]:
        mi, ni = _swat_logical_coord(logical_idx, tiling.m_blocks, tiling.n_blocks)
        sm = _swat_axis_extent(mi, m, tiling.base_m, tiling.m_blocks, m_split, tiling.m_tail_main)
        sn = _swat_axis_extent(ni, n, tiling.base_n, tiling.n_blocks, n_split, tiling.n_tail_main)
        n_off = _swat_axis_origin(ni, tiling.base_n, n_norm, tiling.n_tail_main)
        return sm, sn, n_off

    m_tile, n_tile = max(1, tiling.m_tail_tile), max(1, tiling.n_tail_tile)
    tail_entries = []
    for tail_idx in range(tail_cnt):
        sm, sn, n_off = block_dims(base_round + tail_idx)
        vm, vn = _swat_valid_split(sm, sn, m_tile, n_tile)
        tail_entries.append((base_round + tail_idx, sm, sn, n_off, vm, vn))
    compact_tail = sum(entry[4] * entry[5] for entry in tail_entries)
    if tail_cnt == 0 or compact_tail > block_num:
        m_tile = n_tile = 1
        tail_entries = [(*entry[:4], 1, 1) for entry in tail_entries]

    all_tiles: list[tuple[int, int, int]] = []
    for logical_idx in range(base_round):
        all_tiles.append(block_dims(logical_idx))
    for _, sm, sn, n_off, vm, vn in tail_entries:
        ms = align(ceildiv(sm, m_tile), CUBE_MN_FRACTAL)
        ns = align(ceildiv(sn, n_tile), CUBE_MN_FRACTAL)
        for remain in range(vm * vn):
            m_si, n_si = remain % vm, remain // vm
            tm = min(sm - m_si * ms, ms)
            tn = min(sn - n_si * ns, ns)
            all_tiles.append((tm, tn, n_off + n_si * ns))

    owner_tiles_raw: list[list[tuple[int, int]]] = [[] for _ in range(cores)]
    col_spans: set[tuple[int, int]] = set()
    for tile_idx, (tm, tn, n_off) in enumerate(all_tiles):
        owner_tiles_raw[tile_idx % block_num].append((tm, tn))
        col_spans.add((n_off, tn))
    owner_runs = tuple(
        tuple((shape[0], shape[1], sum(1 for _ in group)) for shape, group in groupby(items))
        for items in owner_tiles_raw
    )
    owner_counts = tuple(len(items) for items in owner_tiles_raw)
    return owner_runs, owner_counts, tuple(sorted(col_spans))


def _swat_buffers(tiling: _SwatTiling) -> tuple[dict[str, int], int]:
    """Per-AIC and per-AIV static buffer footprints for KernelMixWithWeightPrologue."""
    k_al = align(tiling.k_l1, MX_SCALE_GROUP_PAIR)
    scale_k_al = align(tiling.scale_k_l1, MX_SCALE_GROUP_PAIR)
    l1_bytes = (
        tiling.l1_buffer_num * (tiling.base_m + tiling.base_n) * k_al
        + SWAT_L1_DOUBLE_BUFFER * (tiling.base_m + tiling.base_n) * scale_k_al // MX_SCALE_GROUP
    )
    aic_pools = {
        "l1": l1_bytes,
        "l0a": tiling.base_m * tiling.base_k * SWAT_L0_DOUBLE_BUFFER,
        "l0b": tiling.base_n * tiling.base_k * SWAT_L0_DOUBLE_BUFFER,
        "l0c": tiling.base_m * tiling.base_n * SWAT_FP32_BYTES,
    }
    n_al = align(tiling.n_bub_size, CUBE_MN_FRACTAL)
    k_blk = align(tiling.k_bub_size, MX_SCALE_GROUP)
    k_ub_l1 = align(tiling.k_bub_size, MX_SCALE_GROUP_PAIR)
    ub_bytes = tiling.ub_buffer_num * (n_al * k_blk // 2 + n_al * k_ub_l1)
    return aic_pools, ub_bytes


def _swat_panel_work(
    k: int, tm: int, tn: int, k_offset: int, tiling: _SwatTiling,
) -> tuple[int, tuple[tuple[int, int, int, InstructionInventory], ...], tuple[tuple[int, int], ...]]:
    """Source panel work for BlockMmadWeightPrologueMx + KernelMatmulMixWeightPrologue.

    AIC MTE2 loads A, ScaleA, and ScaleB (SCALE_B_SOURCE::AIC).  Each paired AIV
    converts its UpdateUbSliceForCurrentSubBlock K slice via ShiftW4ToW8 and
    publishes only expanded FP8 weights to L1; when k_l1_len <= k_bub_size,
    sub-block 1 skips ConvertCurrentWeightSlice cleanly.
    """
    k_l1_len = min(tiling.k_l1, k - k_offset)
    a_bytes = tm * k_l1_len
    if k_offset % tiling.scale_k_l1 == 0:
        scale_win_k = min(tiling.scale_k_l1, k - k_offset)
        scale_cols = align(scale_win_k, MX_SCALE_GROUP_PAIR) // MX_SCALE_GROUP
        scale_bytes = (tm + tn) * scale_cols
    else:
        scale_bytes = 0
    aic_read_bytes = a_bytes + scale_bytes

    n_align = align(tn, CUBE_MN_FRACTAL)
    aiv_slices: list[tuple[int, int, int, InstructionInventory]] = []
    if k_l1_len <= tiling.k_bub_size:
        k_slices = ((0, k_l1_len),)
    else:
        k_slices = ((0, tiling.k_bub_size), (1, k_l1_len - tiling.k_bub_size))
    for sub, k_slice in k_slices:
        if k_slice <= 0:
            continue
        expanded = n_align * k_slice
        packed = expanded // 2
        aiv_slices.append((sub, packed, expanded, widen(expanded)))

    issued_m = align(tm, CUBE_MN_FRACTAL)
    issued_n = align(tn, CUBE_MN_FRACTAL)
    micros: list[tuple[int, int]] = []
    for k_l0_offset in range(0, k_l1_len, tiling.base_k):
        real_l0_k = min(tiling.base_k, k_l1_len - k_l0_offset)
        issued_k = align(real_l0_k, MX_SCALE_GROUP_PAIR)
        l0_scale_cols = ceildiv(real_l0_k, MX_SCALE_GROUP_PAIR) * SWAT_SCALE_PAIR_COLS
        mte1_bytes = (issued_m + issued_n) * (issued_k + l0_scale_cols)
        micros.append((mte1_bytes, 2 * issued_m * issued_n * issued_k))
    return aic_read_bytes, tuple(aiv_slices), tuple(micros)


def _swat_tile_work(
    k: int, tm: int, tn: int, tiling: _SwatTiling,
) -> tuple[Work, InstructionInventory]:
    total = Work()
    inventory = InstructionInventory({})
    for k_offset in range(0, k, tiling.k_l1):
        aic_read, aiv_slices, micros = _swat_panel_work(k, tm, tn, k_offset, tiling)
        aiv_read = sum(packed for _, packed, _, _ in aiv_slices)
        aiv_write = sum(expanded for _, _, expanded, _ in aiv_slices)
        mte1_total = sum(mte1 for mte1, _ in micros)
        flops_total = sum(flops for _, flops in micros)
        total += Work({"fp8_e4m3": flops_total}, {
            "hbm": aic_read + aiv_read,
            "mte2_aic": aic_read,
            "mte2_aiv": aiv_read,
            "mte3": aiv_write,
            "mte1": mte1_total,
        })
        for _, _, _, inv in aiv_slices:
            total += inv.capacity_work(CLASSES)
            inventory += inv
    out_bytes = 2 * tm * tn
    total += Work(resources={"hbm": out_bytes, "fixpipe": out_bytes})
    return total, inventory


def _swat_ledger(
    k: int, tiling: _SwatTiling,
    owner_runs: tuple[tuple[tuple[int, int, int], ...], ...],
) -> tuple[Work, InstructionInventory]:
    total = Work()
    inventory = InstructionInventory({})
    for schedule in owner_runs[:tiling.used_cores]:
        for tm, tn, repeat in schedule:
            tile_w, tile_inv = _swat_tile_work(k, tm, tn, tiling)
            total += tile_w.scaled(repeat)
            inventory += tile_inv.scaled(repeat)
    return total, inventory


def build(op, geometry):
    """[Cohesive-Pipeline] Plan one shared W4A8 matrix operator from validated hardware geometry."""
    _config, (m, n, k) = config_from_op(op)
    cores, aivs = geometry.compute_units, geometry.resource_units.get("vector", 0)
    if aivs != cores * 2:
        raise ModelingError("shared_w4a8_requires_one_aic_two_aiv_ownership")
    widen_roles = ("reg_issue_slots", "reg_data_writes", "reg_ub_accesses", "reg_ub_stores")
    required = {"mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe", *widen_roles}
    if required - geometry.resource_units.keys():
        raise ModelingError("shared_w4a8_requires_independent_physical_pipes")
    if any(geometry.buffers.get(pool, 0) <= 0 for pool in ("ub", "l1", "l0a", "l0b", "l0c")):
        raise ModelingError("shared_w4a8_source_static_buffers_exceed_device_capacity")

    tiling = _solve_swat_tiling(m, n, k, cores, dict(geometry.buffers))
    aic_pools, ub_bytes = _swat_buffers(tiling)
    owner_runs, counts, col_spans = _swat_owner_schedule(m, n, tiling, cores)
    runs = tuple(len(schedule) for schedule in owner_runs)
    total, inventory = _swat_ledger(k, tiling, owner_runs)
    launched_aivs = 2 * tiling.used_cores

    def owner_program(context, owner):
        ac = context.owned(owner, cores, ("mte2_aic", "mte1", "cube", "fixpipe"))
        vec = [context.owned(owner * 2 + sub, aivs, ("mte2_aiv", "mte3", *widen_roles))
               for sub in (0, 1)]
        allocations = tuple(
            context.allocate(f"{pool}{owner}", pool, size, (owner,))
            for pool, size in aic_pools.items()
        ) + tuple(
            context.allocate(f"ub{owner}_{sub}", "ub", ub_bytes, (2 * owner + sub,))
            for sub in (0, 1)
        )
        schedule = owner_runs[owner]

        def tile(index):
            tm, tn, repeat = schedule[index]
            k_panels = ceildiv(k, tiling.k_l1)

            def panel(panel_idx):
                aic_read, aiv_slices, micros = _swat_panel_work(
                    k, tm, tn, panel_idx * tiling.k_l1, tiling)
                loads = [
                    Stage("a_load", context.bind(Work(resources={
                        "hbm": aic_read * repeat, "mte2_aic": aic_read * repeat,
                    })), placement=ac),
                ]
                for sub, packed, expanded, inv in aiv_slices:
                    loads.extend((
                        Stage(f"b_read{sub}", context.bind(Work(resources={
                            "hbm": packed * repeat, "mte2_aiv": packed * repeat,
                        })), placement=vec[sub]),
                        Stage(f"b_widen{sub}", context.bind(inv.capacity_work(CLASSES).scaled(repeat)),
                              after=(f"b_read{sub}",), placement=vec[sub]),
                        Stage(f"b_publish{sub}", context.bind(Work(resources={
                            "mte3": expanded * repeat,
                        })), after=(f"b_widen{sub}",), placement=vec[sub]),
                    ))
                steps = [
                    (
                        Stage("mte1", context.bind(Work(resources={
                            "mte1": mte1_bytes * repeat,
                        })), placement=ac),
                        Stage("cube", context.bind(Work({
                            "fp8_e4m3": flops * repeat,
                        })), after=("mte1",), placement=ac),
                    )
                    for mte1_bytes, flops in micros
                ]
                return (
                    Stage("load", program=KernelProgram(tuple(loads))),
                    iter((pair, step_idx + 1 == len(steps)) for step_idx, pair in enumerate(steps)),
                    (),
                )

            body = RefillPipeline(k_panels, tiling.l1_buffer_num, SWAT_L0_DOUBLE_BUFFER, panel)
            out_bytes = 2 * tm * tn * repeat
            return (
                Stage("panels", program=body),
                Stage("output", context.bind(Work(resources={
                    "hbm": out_bytes, "fixpipe": out_bytes,
                })), after=("panels",), placement=ac),
            )

        return KernelProgram((
            Stage("reserve", acquire=allocations),
            Stage("tiles", after=("reserve",), program=PipelineProgram(runs[owner], 1, tile)),
            Stage("release", after=("tiles",), free=allocations),
        ))

    def program(context):
        return ParallelProgram(tuple(owner_program(context, owner)
                                     for owner in range(tiling.used_cores)))

    clock_hz = getattr(geometry.compute_arch, "clock_hz", None)
    b8_lanes_per_second = (
        B8_NATIVE_I8_PER_CYCLE_PER_VF * launched_aivs * float(clock_hz)
        if clock_hz is not None and float(clock_hz) > 0.0 else None
    )
    b8 = {
        op_name: {"raw_lanes": count.active_lanes,
                  "bound_us": (
                      count.active_lanes / b8_lanes_per_second * _US_PER_SECOND
                      if b8_lanes_per_second is not None else None
                  )}
        for op_name, count in inventory.counts.items()
        if op_name in {"ShiftRightB8", "ShiftLeftB8", "SelectB8", "AndB8"}
    }
    tile_ms = tuple(sorted({tm for schedule in owner_runs for tm, _tn, _repeat in schedule}))
    return OperatorPlan(total, program, {
        "source": SOURCE,
        "implementation": KIND,
        "coverage": "source_abi_and_swat_tiling_W4A8_matrix_B8_leaf",
        "shared_w4a8": {
            "m": m, "n": n, "k": k,
            "valid_rows": m, "declared_banks": 1,
            "matrix_group_rows": (m,),
            "columns": col_spans,
            "owner_tiles": counts, "owner_runs": runs,
            "swat_tiling": {
                "base_m": tiling.base_m, "base_n": tiling.base_n, "base_k": tiling.base_k,
                "k_l1": tiling.k_l1, "scale_k_l1": tiling.scale_k_l1,
                "n_bub_size": tiling.n_bub_size, "k_bub_size": tiling.k_bub_size,
                "m_blocks": tiling.m_blocks, "n_blocks": tiling.n_blocks,
                "used_cores": tiling.used_cores,
                "m_tail_tile": tiling.m_tail_tile, "n_tail_tile": tiling.n_tail_tile,
                "l1_buffer_num": tiling.l1_buffer_num, "ub_buffer_num": tiling.ub_buffer_num,
                "disable_weight_l2": tiling.disable_weight_l2,
                "ub_bytes_per_aiv": ub_bytes,
                "l1_bytes_per_aic": aic_pools["l1"],
            },
            "logical_flops": 2 * m * n * k,
            "mmad_parameter_m": tile_ms,
            "fixpipe_m": tile_ms,
            "l0a_issue_fractal_m": tuple(sorted({align(tm, CUBE_MN_FRACTAL) for tm in tile_ms})),
            "matrix_issue_basis": "normal arch3510 FP8 Mmad 16x16x32; physical "
                                  "Cube M=16*ceil(m_tile/16); source API/FIX m=valid; "
                                  "matrix ledger only, never elapsed x16",
            "issued_cube_flops": total.matrix.get("fp8_e4m3", 0),
            "b8_typed": {
                "raw_active_lanes": sum(
                    count.active_lanes for op_name, count in inventory.counts.items()
                    if op_name in {"ShiftRightB8", "ShiftLeftB8", "SelectB8", "AndB8"}),
                "per_opcode": b8,
                "native_i8_per_cycle_per_vf": B8_NATIVE_I8_PER_CYCLE_PER_VF,
                "vf_owners": launched_aivs,
                "nominal_clock_hz": clock_hz,
                "prologue": {"CreateMask": SWAT_WIDEN_PROLOGUE_MASK,
                             "DuplicateB8": B8_PROLOGUE_DUPLICATE,
                             "width_elements": B8_WIDEN_BLOCK_ELEMENTS,
                             "once_per_invocation": True},
                "single_port_charge": False,
                "priced_on": "reg_issue_slots_and_ub_ports; no aggregate five-op charge",
            },
            "register_inventory": inventory.to_dict(),
            "register_capacity_work": inventory.capacity_work(CLASSES).resources,
            "register_sources": TABLES,
            "output": "Fixpipe BF16 GM; no SiTU/quant fusion",
            "producer_mode": "swat_weight_prologue_mx",
            "init_mx_bytes": 0,
            "init_mx_basis": "none; QuantBatchMatmulV4WeightQuantMxSwat loads ScaleB on AIC without InitMx",
            "host_binary_status": "swat_host_tiler_and_tail_split_scheduler",
            "unmodeled": [
                "actual_quant_scale_epilogue_and_per_shape_overlap",
                "hardware_cache_hits",
            ],
        },
    })


def recipe():
    return PlannedRecipe(build)
