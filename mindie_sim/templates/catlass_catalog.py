"""Source-pinned CATLASS matrix specializations and schedule enumeration.

[Role] Build concrete K2 kernel specs and expose their integer schedule rules.
[Invariants] Enumerate the declared search space before ranking any candidates.
[Timeouts & Locks] Local finite enumeration only. No shared mutable state.
[Side-Effect Policy] Read-only CPU modeling.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import product
from math import prod

from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import DatapathCapabilitySpec

from .contracts import (
    DIRECT_MATRIX_ARCH_IDS, LEGACY_MATRIX_ARCH_IDS, CatlassKernelSpec, EvgVisitorNode, TemplateSourceRef,
    ceil_div, full_load_a_eligible,
)

CATLASS_REPOSITORY = "Ascend/catlass"
CATLASS_REVISION = "dd03bc0998cbcd67c982e7ab13df6ee15780c6f3"
CATLASS_SOURCE_REFS = (
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/arch/arch.hpp",
        "ae634315dbedfa1f0b66c808caf2763573b801dd0ddba52f2611e1e5a237bca8", "Ascend950", (18, 37)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/arch/cross_core_sync.hpp",
        "5f138acb621c50a28b7aaac748e49148f5ccecc0f3bfed966ed57f40c940db49", "CrossCoreFlagWithReverse", (23, 120)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/catlass.hpp",
        "b19458b6430c303df94c1d2f8c945b3f3b20ff0edb168c945ab0777082cff144", "SIMT_WARP_SIZE", (27, 52)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/dispatch_policy.hpp",
        "0f9c9795b5384f6f211acfaaec9caae62cca99cab3f9c89a8799b337036a38d4", "MmadAtlasA2Pingpong", (38, 42)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/block/block_mmad_pingpong.hpp",
        "d006d6ac2067824f3f285b842290e25cc74799e75488a37160468d822bcb5758", "DispatchStagesGetter", (47, 130)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/kernel/basic_matmul.hpp",
        "180cd954899fc379740e6a5bff77bac25227b02cd3fe4b82fa9f6b52f2f583b3", "BasicMatmul", (20, 120)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/block/block_mmad_pingpong_mutex_tla.hpp",
        "4db59ffb7839192721241b767ee9dd0ef2e3dab51fe22a9e80d46c82f99c5d37", "TOTAL_MUTEX_IDS", (100, 145)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/block/block_mmad_pingpong_mutex_tla.hpp",
        "4db59ffb7839192721241b767ee9dd0ef2e3dab51fe22a9e80d46c82f99c5d37", "ENABLE_L1_RESIDENT", (280, 295)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/block/block_swizzle.hpp",
        "04ac166dd7f3bf0e57f8eb3dce97c26df88d91e4d169850e899b0502dacf3a14", "GemmIdentityBlockSwizzle", (25, 130)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/block/block_swizzle.hpp",
        "04ac166dd7f3bf0e57f8eb3dce97c26df88d91e4d169850e899b0502dacf3a14", "SplitkGemmIdentityBlockSwizzle", (196, 240)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/block/block_swizzle.hpp",
        "04ac166dd7f3bf0e57f8eb3dce97c26df88d91e4d169850e899b0502dacf3a14", "StreamkGemmIdentityBlockSwizzle", (870, 1025)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/kernel/dynamic_padding_streamk_matmul.hpp",
        "27b07884d4c3f19a5c37d7996bb62dc7c195ef525e97410adc6e5ab5197d4122", "DynamicPaddingStreamkMatmul", (170, 470)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/kernel/streamk_matmul_tla.hpp",
        "bf6baaae8bdf969a6c21519da997d9c505d98add5ae187b70e0dc9a65e5a32f4", "StreamkMatmulTla", (25, 120)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/kernel/dynamic_padding_multi_core_splitk_matmul.hpp",
        "168ec2485cc2dc29e0d41db053d052e7c1b1505a2b8288bf1ed5152b97a74ed3", "DynamicPaddingMultiCoreSplitkMatmul", (25, 160)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/epilogue/block/block_epilogue_visitor.hpp",
        "4a2ca2b0a0e3b5ca7985a0be23d13dc19f2edc0d745d2df02e491729c169c279", "GetMmadUbTensor", (26, 130)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/epilogue/fusion/operations.hpp",
        "072617fe53081d8f98ac2b3ce7c5b38c52c1125bc01d7203e10effa877723a3d", "Exp", (20, 151)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "examples/64_ascend950_matmul_evg/matmul_evg_add_ub.cpp",
        "975d11bb5ed4b38a978588e1d23edb56f784b573da84085eb1568f0af85bdfac", "EpilogueVisitor", (120, 160)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/kernel/grouped_matmul_slice_m.hpp",
        "1506d5d49e4169ea98b19ca434cca03ebd6d14d252d66e4a046e6fc56aabce56", "GroupedMatmulSliceM", (24, 195)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/kernel/grouped_mx_matmul_slice_m_swiglu_mx_quant_tla.hpp",
        "18b901ed463bc01b4004af90f743adc1094ab578c399586c6aa54ddce42a8a77", "GroupedMxMatmulSliceMSwigluMxQuantTla", (28, 95)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/block/block_mmad_mx_tla.hpp",
        "a946988a9f3a049e8702d01e1c8453c047cf6ab9f4f2fbe7ead94a490dfa974a", "L1_SCALE_FACTOR_K", (30, 120)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/block/block_mmad_mx_a8w4.hpp",
        "f3b30b3de704bbe780bfd2fcb786f59b827345576b53cc64d6917c01e5fb45bb", "BlockMmadA8W4Mx", (29, 110)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "examples/59_ascend950_a8w4_mx_matmul/a8w4_mx.cpp",
        "6c04e637cf8cb8d6b77c7a609ebed44816aec39db6171706d4bc4c2f9900396a", "ElementPrologueB", (50, 200)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "python/tla_dsl/catlass/core_api.py",
        "16a7bbdeb90ed3cd3e62dbcf80172decef0c88cd788a0f672cd7791bb242b324", "thread_block_dim", (8064, 8128)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/kernel/matmul_gather_scatter_simt.hpp",
        "c95d5a3d1e7591d3289e42ea26c850b5b0081a9d687eea89cbdda542ef8c5040", "RunMatmulGatherScatterSimt", (80, 140)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/kernel/matmul_gather_scatter_full_load.hpp",
        "0495ecca9fd3be283b4e9abcc625daaafbfb8ff647b936d54a8cd6e134d23d3d", "MatmulGatherScatterFullLoadKernelBase", (81, 138)),
    TemplateSourceRef(CATLASS_REPOSITORY, CATLASS_REVISION, "include/catlass/gemm/kernel/grouped_mx_matmul_finalize_routing_tla.hpp",
        "14541031fe54e63ce441777442a70024b9b53dba357ca0d551966e4a99f69757", "GetWorkspaceSize", (150, 165)),
)
_CATLASS_SYMBOLS = {
    "catlass.basic_pingpong": ("MmadAtlasA2Pingpong", "DispatchStagesGetter", "BasicMatmul"),
    "catlass.mutex_block_mmad": ("TOTAL_MUTEX_IDS", "ENABLE_L1_RESIDENT"),
    "catlass.streamk": ("StreamkGemmIdentityBlockSwizzle", "DynamicPaddingStreamkMatmul", "StreamkMatmulTla", "CrossCoreFlagWithReverse"),
    "catlass.multi_core_splitk": ("SplitkGemmIdentityBlockSwizzle", "DynamicPaddingMultiCoreSplitkMatmul", "CrossCoreFlagWithReverse"),
    "catlass.swizzle": ("GemmIdentityBlockSwizzle", "BasicMatmul"),
    "catlass.evg": ("GetMmadUbTensor", "Exp", "EpilogueVisitor"),
    "catlass.grouped_slice_m": ("GroupedMatmulSliceM",),
    "catlass.grouped_mx_swiglu_mx_quant": ("GroupedMatmulSliceM", "GroupedMxMatmulSliceMSwigluMxQuantTla", "L1_SCALE_FACTOR_K"),
    "catlass.mx_matmul": ("L1_SCALE_FACTOR_K", "BlockMmadA8W4Mx", "ElementPrologueB"),
    "catlass.a5_simt": ("SIMT_WARP_SIZE", "thread_block_dim", "RunMatmulGatherScatterSimt", "GetWorkspaceSize"),
    "catlass.full_load_a_gather_scatter": ("MatmulGatherScatterFullLoadKernelBase",),
}
CATLASS_TEMPLATE_IDS = tuple(_CATLASS_SYMBOLS)
MAX_MUTEX_IDS = 28
MX_SCALE_GROUP = 32
A8W4_L1_SCALE_FACTOR_K = 16
SIMT_TLA_MAX_THREADS = 2048
SIMT_CATLASS_MAX_THREADS = 1024
GATHER_SCATTER_SIMT_UB_BYTES = 64 * 1024
GATHER_SCATTER_FULL_LOAD_A = "MatmulGatherScatterFullLoadA"
GATHER_SCATTER_POLICIES = frozenset({"MatmulGatherScatterSimt", GATHER_SCATTER_FULL_LOAD_A})
GATHER_SCATTER_AIV_MAX_K = 768
STREAMK_TLA_MIN_WORKSPACE = 2 * 1024 * 1024
STREAMK_PARTIAL_SLOTS_PER_CORE = 2
DEFAULT_L1_TILES = ((128, 128, 128), (64, 128, 128), (64, 64, 64))
DEFAULT_L0_PANEL_K = 64
DEFAULT_SWIZZLE_OFFSETS = (1, 2, 3, 4, 6, 8, 12, 16)
DEFAULT_SIMT_BLOCK = (128, 1, 1)
_MATRIX_SHAPE_RANK = 3
_RING_COUNT = 5
_DEFAULT_STAGE_DEPTHS = (2, 2, 2, 2, 1)
_MUTEX_STAGE_DEPTHS = (2, 2, 2, 2, 2)
_DTYPE_ALIASES = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16"}


def dtype_name(dtype: str) -> str:
    return _DTYPE_ALIASES.get(dtype.lower(), dtype.lower())


def packed_bytes(elements: int, dtype: str) -> int:
    # MX scale storage has its own allocation and transfer ledger.
    return ceil_div(elements * int(2 * dtype_bytes(dtype_name(dtype))), 2)


def unmodeled(template: str, reason: str) -> UnmodeledCapabilityError:
    return UnmodeledCapabilityError(
        capability=template, entry="template_catalog", code=reason,
        message=f"{template}: {reason}",
    )


@dataclass(frozen=True, slots=True)
class CatlassProblem:
    m: int
    n: int
    k: int
    dtype_a: str = "bf16"
    dtype_b: str = "bf16"
    dtype_c: str = "bf16"
    dtype_acc: str = "fp32"
    groups: tuple[int, ...] = ()
    a_id: str = "a"
    b_id: str = "b"
    c_id: str = "c"
    has_bias: bool = False
    mx_mode: str | None = None
    indices: tuple[int, ...] | None = None
    physical_m: int | None = None
    indices_id: str = "indices"

    def __post_init__(self) -> None:
        if any(type(n) is not int or n < 1 for n in (self.m, self.n, self.k)):
            raise ModelingError("catlass_problem_shape_invalid")
        if self.groups and (any(type(n) is not int or n < 0 for n in self.groups) or sum(self.groups) != self.m):
            raise ModelingError("catlass_group_rows_invalid")
        if len({self.a_id, self.b_id, self.c_id}) != _MATRIX_SHAPE_RANK or not all((self.a_id, self.b_id, self.c_id)):
            raise ModelingError("catlass_operand_id_invalid")
        if self.physical_m is not None and (type(self.physical_m) is not int or self.physical_m < self.m):
            raise ModelingError("catlass_physical_row_count_invalid")
        if self.indices is not None and (
            len(self.indices) != self.m or any(type(i) is not int or i < 0 for i in self.indices)
            or (self.physical_m is not None and max(self.indices) >= self.physical_m)
            or not self.indices_id or self.indices_id in {self.a_id, self.b_id, self.c_id}
        ):
            raise ModelingError("catlass_gather_scatter_indices_invalid")

    @property
    def output_rows(self) -> int:
        return self.physical_m or max(self.m, max(self.indices or (), default=-1) + 1)


def matrix_problem(problem: CatlassProblem | Mapping) -> CatlassProblem:
    return CatlassProblem(**problem) if isinstance(problem, Mapping) else problem


def should_elide_l1_reload(prev_addr, prev_coord, next_addr, next_coord, enable_l1_resident) -> bool:
    return bool(enable_l1_resident and prev_addr == next_addr and prev_coord == next_coord)


def swizzle_block_coord(task_idx: int, gm: int, gn: int, direction: str, offset: int) -> tuple[int, int]:
    if min(gm, gn, offset) < 1 or task_idx < 0 or direction not in {"Zn", "Nz"}:
        raise ModelingError("catlass_swizzle_geometry_invalid")
    major, minor = (gm, gn) if direction == "Zn" else (gn, gm)
    band, inner = divmod(task_idx % (gm * gn), offset * minor)
    width = min(offset, major - band * offset)
    orthogonal, strip = divmod(inner, width)
    if band % 2:
        orthogonal = minor - 1 - orthogonal
    along = band * offset + strip
    return (along, orthogonal) if direction == "Zn" else (orthogonal, along)


def wave_band_counts(gm: int, gn: int, c_act: int, direction: str, offset: int) -> tuple[int, int]:
    coords = (swizzle_block_coord(i, gm, gn, direction, offset) for i in range(min(c_act, gm * gn)))
    rows, cols = set(), set()
    for m, n in coords:
        rows.add(m)
        cols.add(n)
    return len(rows), len(cols)


@dataclass(frozen=True, slots=True)
class StreamKPartition:
    b_sk: int
    b_norm: int
    k_base: int
    k_rem: int
    intervals: tuple[tuple[int, int], ...]
    partial_counts: tuple[int, ...]

    @property
    def partial_count(self) -> int:
        return sum(self.partial_counts)


def streamk_partition(b_tot: int, g_k: int, c_aic: int) -> StreamKPartition:
    if b_tot < 0 or min(g_k, c_aic) < 1:
        raise ModelingError("catlass_streamk_geometry_invalid")
    b_sk = b_tot % c_aic
    k_base, k_rem = divmod(b_sk * g_k, c_aic)
    intervals, counts = [], []
    start = 0
    for core in range(c_aic):
        end = start + k_base + (core < k_rem)
        intervals.append((start, end))
        counts.append(0 if end == start else (end - 1) // g_k - start // g_k + 1)
        start = end
    return StreamKPartition(b_sk, b_tot - b_sk, k_base, k_rem, tuple(intervals), tuple(counts))


def grouped_start_cores(groups: Iterable[int], n: int, tm: int, tn: int, cores: int):
    if min(n, tm, tn, cores) < 1:
        raise ModelingError("catlass_grouped_geometry_invalid")
    start = 0
    for index, rows in enumerate(groups):
        if rows < 0:
            raise ModelingError("catlass_group_rows_invalid")
        if rows:
            loops = ceil_div(rows, tm) * ceil_div(n, tn)
            yield index, start, loops
            start = (start + loops) % cores


def mx_scale_bytes(m: int, k: int, factor: int) -> int:
    return (m * k // MX_SCALE_GROUP) * factor


def validate_catlass_spec(spec: CatlassKernelSpec, p: CatlassProblem, datapath: DatapathCapabilitySpec) -> None:
    name = spec.template_id
    if name not in _CATLASS_SYMBOLS or datapath.arch_id not in LEGACY_MATRIX_ARCH_IDS | DIRECT_MATRIX_ARCH_IDS:
        raise unmodeled(name, "catlass_template_or_architecture_unsupported")
    modern = datapath.arch_id in DIRECT_MATRIX_ARCH_IDS
    if spec.arch_tag != ("Ascend950" if modern else "AtlasA2"):
        raise unmodeled(name, "catlass_spec_architecture_mismatch")
    if not (1 <= spec.aic_budget <= datapath.aic_count and 0 <= spec.aiv_budget <= datapath.aiv_count):
        raise ModelingError("catlass_core_budget_invalid")
    if len(spec.stages) != _RING_COUNT or min(spec.stages) < 1:
        raise ModelingError("catlass_ring_depths_invalid")
    if any(len(s) != _MATRIX_SHAPE_RANK or min(s) < 1 for s in (spec.l1_tile_shape, spec.l0_tile_shape)):
        raise ModelingError("catlass_tile_shape_invalid")
    if spec.l1_tile_shape[:2] != spec.l0_tile_shape[:2] or spec.l0_tile_shape[-1] > spec.l1_tile_shape[-1]:
        raise ModelingError("catlass_l1_l0_tile_mismatch")
    if spec.template_id == "catlass.mutex_block_mmad" and spec.dispatch_policy != "MmadPingpongMutex":
        raise ModelingError("catlass_mutex_dispatch_policy_required")
    if spec.template_id == "catlass.a5_simt" and (spec.simt_thread_block_dim is None or spec.dispatch_policy not in {
        "MatmulGatherScatterSimt", "tla_dsl", "GroupedMxMatmulFinalizeRoutingTla",
    }):
        raise ModelingError("catlass_simt_dispatch_policy_required")
    indexed = spec.dispatch_policy in GATHER_SCATTER_POLICIES
    if (p.indices is not None or p.physical_m is not None) and not indexed:
        raise ModelingError("catlass_indexed_operands_require_gather_scatter")
    if indexed:
        if p.has_bias or spec.has_bias or p.groups:
            raise ModelingError("catlass_gather_scatter_requires_single_unbiased_product")
        for operands in (p, spec):
            if any(dtype_name(dtype) != "fp16" for dtype in (operands.dtype_a, operands.dtype_b, operands.dtype_c)):
                raise ModelingError("catlass_gather_scatter_requires_fp16")
            if dtype_name(operands.dtype_acc) != "fp32":
                raise ModelingError("catlass_gather_scatter_requires_fp32_accumulator")
        if (spec.layout_a, spec.layout_b, spec.layout_c) != ("RowMajor",) * _MATRIX_SHAPE_RANK:
            raise ModelingError("catlass_gather_scatter_requires_row_major")
    if spec.dispatch_policy == GATHER_SCATTER_FULL_LOAD_A and (
        name != "catlass.full_load_a_gather_scatter" or spec.simt_thread_block_dim is not None
        or not full_load_a_eligible(p, datapath) or spec.l1_tile_shape[-1] != p.k or not spec.aiv_budget
    ):
        raise ModelingError("catlass_full_load_a_geometry_invalid")
    if spec.dispatch_policy == "MatmulGatherScatterSimt" and name != "catlass.a5_simt":
        raise ModelingError("catlass_simt_dispatch_policy_required")
    if spec.template_id in {"catlass.mx_matmul", "catlass.grouped_mx_swiglu_mx_quant"} and not spec.mx_scale_group:
        raise ModelingError("catlass_scale_specialization_required")
    if spec.enable_unit_flag and spec.stages[-1] != 1:
        raise ModelingError("catlass_unit_flag_requires_single_l0c_stage")
    if spec.dispatch_policy == "MmadPingpongMutex" and sum(spec.stages) + 2 * spec.has_bias > MAX_MUTEX_IDS:
        raise ModelingError("catlass_mutex_ids_exhausted")
    if name == "catlass.basic_pingpong" and (len(set(spec.stages[:-1])) != 1 or spec.stages[-1] != 1):
        raise ModelingError("catlass_pingpong_requires_common_stage_count")
    if spec.swizzle_offset < 1 or spec.swizzle_direction not in {"Zn", "Nz"}:
        raise ModelingError("catlass_swizzle_geometry_invalid")
    if spec.scheduler == "multi_core_splitk":
        g_k = ceil_div(p.k, spec.l1_tile_shape[-1])
        if not 2 <= spec.splitk_factor <= min(g_k, spec.aic_budget):
            raise ModelingError("catlass_splitk_factor_invalid")
    elif spec.splitk_factor != 1:
        raise ModelingError("catlass_splitk_factor_requires_splitk_scheduler")
    if bool(p.groups) != (spec.scheduler == "grouped_slice_m"):
        raise ModelingError("catlass_grouped_problem_requires_grouped_scheduler")
    if spec.epilogue_kind != "none" or spec.scheduler in {"streamk", "multi_core_splitk"}:
        if not spec.aiv_budget:
            raise ModelingError("catlass_epilogue_requires_aiv")
    if spec.epilogue_kind in {"evg_ub", "grouped_swiglu_mx_quant"} and not datapath.has_l0c_to_ub:
        raise unmodeled(name, "catlass_direct_epilogue_requires_l0c_to_ub")
    if spec.epilogue_kind == "grouped_swiglu_mx_quant":
        if p.n % 2 or (p.n // 2) % spec.l1_tile_shape[1]:
            raise ModelingError("catlass_swiglu_gate_up_tiles_misaligned")
        if not spec.mx_scale_group:
            raise ModelingError("catlass_grouped_swiglu_requires_mx")
    if spec.mx_scale_group:
        arch = datapath.compute_arch.declared
        mode = p.mx_mode or dtype_name(spec.dtype_a)
        if arch is None or not arch.supports_native("matrix", mode):
            raise unmodeled(name, f"catlass_native_mx_mode_unsupported:{mode}")
        a8w4 = spec.prologue_b == "DecompressFp4ToFp8"
        if a8w4 and not datapath.has_ub_to_l1:
            raise unmodeled(name, "catlass_a8w4_requires_ub_to_l1")
        if a8w4 and (packed_bytes(2, spec.dtype_a) != 2 or packed_bytes(2, spec.dtype_b) != 1):
            raise ModelingError("catlass_a8w4_operand_representation_invalid")
        if spec.mx_scale_group != MX_SCALE_GROUP or spec.l1_scale_factor_k != (A8W4_L1_SCALE_FACTOR_K if a8w4 else 1):
            raise ModelingError("catlass_mx_scale_geometry_invalid")
    if spec.simt_thread_block_dim is not None:
        if not modern or datapath.simt_ub_carveout_bytes <= 0:
            raise unmodeled(name, "catlass_simt_capability_missing")
        dims = spec.simt_thread_block_dim
        limit = SIMT_CATLASS_MAX_THREADS if spec.dispatch_policy == "MatmulGatherScatterSimt" else SIMT_TLA_MAX_THREADS
        if len(dims) != _MATRIX_SHAPE_RANK or min(dims) < 1 or prod(dims) > limit or not spec.aiv_budget:
            raise ModelingError("catlass_simt_thread_block_invalid")


def _workspace_bytes(template_id, p, tile, aic, split, policy, epilogue):
    m, n, _ = tile
    acc = packed_bytes(p.m * p.n, p.dtype_acc)
    if epilogue == "evg_gm":
        return acc
    if epilogue == "finalize_routing":
        return packed_bytes(m * n * aic, p.dtype_c)
    if template_id == "catlass.multi_core_splitk":
        return acc * split
    if template_id == "catlass.streamk":
        slots = STREAMK_PARTIAL_SLOTS_PER_CORE * aic
        partials = packed_bytes(m * n * slots, p.dtype_acc)
        return max(partials, STREAMK_TLA_MIN_WORKSPACE) if policy == "StreamkMatmulTla" else partials
    return 0


def enumerate_catlass_specs(
    problem: CatlassProblem | Mapping, datapath: DatapathCapabilitySpec, *,
    template_ids: Iterable[str] = CATLASS_TEMPLATE_IDS,
    l1_tiles: Iterable[tuple[int, int, int]] = DEFAULT_L1_TILES,
    l0_tiles: Iterable[tuple[int, int, int]] | None = None,
    stage_options: Iterable[tuple[int, int, int, int, int]] | None = None,
    swizzle_offsets: Iterable[int] = DEFAULT_SWIZZLE_OFFSETS,
    evg_tree: tuple[EvgVisitorNode, ...] = (),
    simt_variants: Iterable[str] = ("MatmulGatherScatterSimt", "tla_dsl", "GroupedMxMatmulFinalizeRoutingTla"),
    simt_thread_blocks: Iterable[tuple[int, int, int]] = (DEFAULT_SIMT_BLOCK,),
    aic_budget: int | None = None, aiv_budget: int | None = None,
    enable_unit_flag: bool = False, enable_l1_resident: bool = False,
) -> tuple[CatlassKernelSpec, ...]:
    """Return every feasible specialization of the caller's finite search space."""
    from .catlass_lower import solve_catlass_tile

    p = matrix_problem(problem)
    aic = datapath.aic_count if aic_budget is None else aic_budget
    aiv = datapath.aiv_count if aiv_budget is None else aiv_budget
    if not (1 <= aic <= datapath.aic_count and 0 <= aiv <= datapath.aiv_count):
        raise ModelingError("catlass_core_budget_invalid")
    stage_options = None if stage_options is None else tuple(stage_options)
    swizzle_offsets = tuple(swizzle_offsets)
    simt_variants, simt_thread_blocks = tuple(simt_variants), tuple(simt_thread_blocks)
    l0_tiles = None if l0_tiles is None else tuple(l0_tiles)
    tiles = tuple(l1_tiles)
    modern = datapath.arch_id in DIRECT_MATRIX_ARCH_IDS
    output = {}
    for name in template_ids:
        if name not in _CATLASS_SYMBOLS:
            raise ModelingError(f"catlass_template_unknown:{name}")
        grouped = name.startswith("catlass.grouped_")
        mx = name in {"catlass.mx_matmul", "catlass.grouped_mx_swiglu_mx_quant"}
        if bool(p.groups) != grouped or (name == "catlass.evg" and not evg_tree) or (mx and not p.mx_mode and not dtype_name(p.dtype_a).startswith(("mxfp", "mx_fp"))):
            continue
        scheduler = name.removeprefix("catlass.") if name in {"catlass.streamk", "catlass.multi_core_splitk"} else (
            "grouped_slice_m" if grouped else "dataparallel"
        )
        swizzles = tuple(product(("Zn", "Nz"), swizzle_offsets)) if name == "catlass.swizzle" else (("Zn", 1),)
        simt = name == "catlass.a5_simt"
        policies = simt_variants if simt else ({
            "catlass.full_load_a_gather_scatter": GATHER_SCATTER_FULL_LOAD_A,
            "catlass.mutex_block_mmad": "MmadPingpongMutex",
            "catlass.streamk": "StreamkMatmulTla" if modern else "MmadAtlasA2Streamk",
            "catlass.mx_matmul": "MmadA8W4Mx" if p.mx_mode == "a8w4_mx" else "MmadMx",
            "catlass.grouped_mx_swiglu_mx_quant": "MmadMx",
        }.get(name, "MmadPingpong" if modern else "MmadAtlasA2Pingpong"),)
        default_stages = (_MUTEX_STAGE_DEPTHS if name == "catlass.mutex_block_mmad" and not enable_unit_flag
                          else _DEFAULT_STAGE_DEPTHS)
        for l1, raw_stages, (direction, offset), policy in product(
            tiles, (default_stages,) if stage_options is None else stage_options, swizzles, policies,
        ):
            tm, tn, tk = l1
            if min(l1) < 1:
                raise ModelingError("catlass_tile_shape_invalid")
            if policy == GATHER_SCATTER_FULL_LOAD_A:
                tk = p.k
                l1 = (tm, tn, tk)
            micro_tiles = l0_tiles if l0_tiles is not None else ((tm, tn, min(tk, DEFAULT_L0_PANEL_K)),)
            splits = range(2, min(aic, ceil_div(p.k, tk)) + 1) if scheduler == "multi_core_splitk" else (1,)
            epilogues = {
                "catlass.evg": ("evg_gm", "evg_ub") if datapath.has_l0c_to_ub else ("evg_gm",),
                "catlass.grouped_mx_swiglu_mx_quant": ("grouped_swiglu_mx_quant",),
            }.get(name, ("finalize_routing",) if policy == "GroupedMxMatmulFinalizeRoutingTla" else ("none",))
            stages = raw_stages
            if policy == GATHER_SCATTER_FULL_LOAD_A:
                stages = (1, *raw_stages[1:])
            if name == "catlass.basic_pingpong":
                depth = min(raw_stages[:-1])
                stages = (depth, depth, depth, depth, 1)
            a8w4 = mx and p.mx_mode == "a8w4_mx"
            for l0, split, epilogue, block in product(micro_tiles, splits, epilogues, simt_thread_blocks if simt else (None,)):
                refs = tuple(ref for ref in CATLASS_SOURCE_REFS if ref.symbol in (*_CATLASS_SYMBOLS[name], "Ascend950"))
                spec = CatlassKernelSpec(
                    name, "Ascend950" if modern else "AtlasA2", policy,
                    p.dtype_a, p.dtype_b, p.dtype_c, p.dtype_acc,
                    "RowMajor", "ColumnMajor" if a8w4 else "RowMajor", "RowMajor",
                    None, "DecompressFp4ToFp8" if a8w4 else None,
                    MX_SCALE_GROUP if mx else 0, "E8M0_NZ_G32" if mx else None,
                    (A8W4_L1_SCALE_FACTOR_K if a8w4 else 1) if mx else 0,
                    l1, l0, stages, p.has_bias, enable_unit_flag, enable_l1_resident, scheduler,
                    split, offset, direction,
                    "disable" if grouped and all(rows <= tm for rows in p.groups) else "normal",
                    epilogue, evg_tree if name == "catlass.evg" else (),
                    _workspace_bytes(name, p, l1, aic, split, policy, epilogue),
                    block, aic, aiv, refs,
                )
                try:
                    solve_catlass_tile(spec, p, datapath)
                except ModelingError:
                    continue
                output[spec] = None
    return tuple(output)
