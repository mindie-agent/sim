"""Immutable template specializations and whole-step accounting contracts.

[Role] Define the public catalog, compiler and certificate interface records.
[Invariants] Provenance pins blobs and symbols. Byte ranges are half-open.
[Timeouts & Locks] Validation and composition use local data without locks.
[Side-Effect Policy] No device, repository or filesystem mutation.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from graphlib import CycleError, TopologicalSorter
from math import isfinite, lcm, prod
from typing import Any, Iterable, Literal

from mindie_sim.prediction.fusion.ports import input_port_entries
from mindie_sim.prediction.program.stages import KernelProgram
from mindie_sim.prediction.runtime.memory_physics import (
    AddressRun, SectorCoverage, _MAX_EXPLICIT_SECTOR_INTERVALS, _merge_byte_intervals,
)
from mindie_sim.prediction.runtime.l2_residency import (
    _coalesce_runs, _coverage_runs, _partition_runs, _range_run, _residency_runs, _run_end,
)
from mindie_sim.prediction.runtime.tile_solver import AlgorithmKind, TilePipelineSpec, TileSolution
from mindie_sim.schema.abi import SOC_ARCHS, SocArch
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

COMMIT_HEX_LENGTH = 40
SHA256_HEX_LENGTH = 64
LEGACY_MATRIX_ARCH_IDS = frozenset({"npu_arch_2201", "arch22"})
DIRECT_MATRIX_ARCH_IDS = frozenset({"npu_arch_3510", "arch35"})
FULL_LOAD_A_MAX_K = 2048
FP16_ELEMENTS_PER_BLOCK = 16
TILELANG_BACKENDS = frozenset({"TileLang", "tilelang_simt", "tilelang_simd"})

TileLangApiTier = Literal["tilelang_simt", "tilelang_simd"]
TileLangDispatchMode = Literal["aclgraph", "eager_prealloc", "eager_callee_alloc"]
TileLangDmaPattern = Literal["rw11", "rw21"]
TileLangCubeEpilogue = Literal["fixpipe", "dual_copy", "atomic_add"]
FuseRegionBackend = Literal[
    "catlass_evg", "dsl_mix", "dsl_vec", "tla_simt", "tilelang_simt", "tilelang_simd",
]
LoweringTierKind = Literal[
    "tier1_shipping_abi", "tilelang_simt", "tilelang_simd", "ascendc", "catlass",
]
LOWERING_TIER_KINDS: frozenset[str] = frozenset({
    "tier1_shipping_abi", "tilelang_simt", "tilelang_simd", "ascendc", "catlass",
})

EvgKind = Literal[
    "acc_load", "aux_load", "row_broadcast", "col_broadcast", "exp", "relu",
    "sqrt", "rsqrt_fast", "leaky_relu", "silu", "cast", "mul", "muls", "add",
    "aux_store",
]
_EVG_ARITY = {
    "acc_load": 0, "aux_load": 0, "row_broadcast": 0, "col_broadcast": 0,
    "exp": 1, "relu": 1, "sqrt": 1, "rsqrt_fast": 1, "leaky_relu": 1,
    "silu": 1, "cast": 1, "mul": 2, "muls": 1, "add": 2, "aux_store": 1,
}
_EVG_TENSOR_NODES = {"aux_load", "row_broadcast", "col_broadcast", "aux_store"}


def _hex(value: str, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and bool(re.fullmatch("[0-9a-fA-F]+", value))


FP32_BYTES = 4
BF16_BYTES = 2
KDA_HEAD_DIM = 128
KDA_STATE_HALF = KDA_HEAD_DIM // 2
COMMIT_SMALL_HEADS = 16
COMMIT_MEDIUM_HEADS = 32
COMMIT_SMALL_ROWS = 32

ATTN_RES_UB_BYTES = 240 * 1024
ATTN_RES_MAX_TILE_T = 4095
ATTN_RES_FP32_ALIGN = 8
ATTN_RES_BF16_ALIGN = 16
ATTN_RES_TILE_FP32_ROWS = 2
ATTN_RES_STAT_COUNT = 3

def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def attn_res_ub_footprint(d, tile_t, depth):
    if min(d, tile_t, depth) < 1:
        raise ModelingError("attn_res_tile_invalid")
    s32 = ceil_div(d, ATTN_RES_FP32_ALIGN) * ATTN_RES_FP32_ALIGN
    s16 = ceil_div(d, ATTN_RES_BF16_ALIGN) * ATTN_RES_BF16_ALIGN
    s_stat = ceil_div(tile_t, ATTN_RES_FP32_ALIGN) * ATTN_RES_FP32_ALIGN
    return FP32_BYTES * s32 + BF16_BYTES * s16 + depth * (
        tile_t * (ATTN_RES_TILE_FP32_ROWS * FP32_BYTES * s32 + BF16_BYTES * s16)
        + ATTN_RES_STAT_COUNT * FP32_BYTES * s_stat
    )


def attn_res_tiling(tokens: int, aiv: int, ub_bytes: int, hidden_size: int) -> tuple[int, int, int, int]:
    """Compute AttnRes UB tiling ``(active_aiv, rows_per_aiv, tile_rows, depth)`` from explicit ``hidden_size``."""
    if any(type(v) is not int or v < 1 for v in (tokens, aiv, ub_bytes, hidden_size)):
        raise ModelingError("attn_res_problem_invalid")
    rows = ceil_div(tokens, aiv)
    d = hidden_size
    row_bytes = d * (ATTN_RES_TILE_FP32_ROWS * FP32_BYTES + BF16_BYTES) + ATTN_RES_STAT_COUNT * FP32_BYTES
    tile = min(ceil_div(rows, 2), ATTN_RES_MAX_TILE_T, (ub_bytes - d * (FP32_BYTES + BF16_BYTES)) // (2 * row_bytes))
    while tile > 0 and attn_res_ub_footprint(d, tile, 2) > ub_bytes:
        tile -= 1
    if tile < 1:
        raise ModelingError("attn_res_double_buffered_row_does_not_fit")
    count = ceil_div(rows, tile)
    return ceil_div(tokens, rows), rows, ceil_div(rows, count), 1 if count == 1 else 2


def flash_kda_rounds(chunks, heads, slots, max_group):
    batches_per_set = slots // heads
    if min(batches_per_set, max_group) < 1:
        raise ModelingError("flash_kda_workspace_group_invalid")
    for first in range(0, len(chunks), batches_per_set):
        remaining = list(chunks[first:first + batches_per_set])
        while max(remaining) > 0:
            low, high = 1, min(max_group, max(remaining), batches_per_set)
            while low < high:
                mid = (low + high + 1) // 2
                if sum(min(count, mid) for count in remaining) <= batches_per_set:
                    low = mid
                else:
                    high = mid - 1
            take = tuple(min(count, low) for count in remaining)
            yield first, take
            remaining = [left - used for left, used in zip(remaining, take)]


def commit_replay_bytes(row_partitions, accepted=1, d=KDA_HEAD_DIM):
    if not 1 <= row_partitions <= d or accepted < 0 or d != KDA_HEAD_DIM:
        raise ModelingError("commit_kda_replay_geometry_invalid")
    return accepted * d * (FP32_BYTES + (BF16_BYTES + FP32_BYTES) * row_partitions)


def commit_rows(p, aiv_budget):
    """Rows assigned to each AIV lane, including split halves in the large regime."""
    heads = len(p.accepted) * p.heads
    if not heads:
        return ()
    if p.row_partitions is not None or heads <= COMMIT_MEDIUM_HEADS:
        partitions = p.row_partitions
        if partitions is None:
            partitions = KDA_HEAD_DIM // (COMMIT_SMALL_ROWS if heads <= COMMIT_SMALL_HEADS else KDA_STATE_HALF)
        if partitions < 1 or KDA_HEAD_DIM % partitions:
            raise ModelingError("commit_kda_row_partition_invalid")
        rows = KDA_HEAD_DIM // partitions
        lanes = min(heads * partitions, aiv_budget)
        return tuple(tuple((item // partitions, item % partitions * rows, rows)
                           for item in range(lane, heads * partitions, lanes))
                     for lane in range(lanes))
    lanes = min(heads, aiv_budget)
    base, extra = divmod(2 * heads, lanes)
    full_heads = lanes * (base // 2) + extra * (ceil_div(base, 2) - base // 2)
    schedules = []
    start = 0
    for lane in range(lanes):
        full = ceil_div(base, 2) if lane < extra else base // 2
        items = [(head, 0, KDA_HEAD_DIM) for head in range(start, start + full)]
        start += full
        if base % 2 == 0 and lane < extra:
            items.append((full_heads + lane // 2, lane % 2 * KDA_STATE_HALF, KDA_STATE_HALF))
        if base % 2 and lane >= extra:
            half = lane - extra
            items.append((full_heads + half // 2, half % 2 * KDA_STATE_HALF, KDA_STATE_HALF))
        schedules.append(tuple(items))
    return tuple(schedules)


@dataclass(frozen=True, slots=True)
class TemplateSourceRef:
    repository: str
    revision: str
    relative_path: str
    source_sha256: str
    symbol: str
    line_range: tuple[int, int]

    def __post_init__(self) -> None:
        path = self.relative_path
        if not isinstance(self.repository, str) or not self.repository.strip() or not _hex(self.revision, COMMIT_HEX_LENGTH):
            raise ModelingError("template_source_revision_invalid")
        if not isinstance(path, str) or not path or set(path.split("/")) & {"", ".", ".."} or any(c in path for c in ("\\", ":", "\x00")):
            raise ModelingError("template_source_path_invalid")
        if not _hex(self.source_sha256, SHA256_HEX_LENGTH) or not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ModelingError("template_source_blob_or_symbol_invalid")
        if (
            type(self.line_range) is not tuple or len(self.line_range) != 2
            or any(type(x) is not int for x in self.line_range)
            or not 1 <= self.line_range[0] <= self.line_range[1]
        ):
            raise ModelingError("template_source_line_range_invalid")

    def to_source_ref(self) -> SourceRef:
        return SourceRef(self.repository, self.revision, self.relative_path, self.source_sha256)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvgVisitorNode:
    node_id: str
    kind: EvgKind
    dtype: str
    layout: str = "RowMajor"
    inputs: tuple[str, ...] = ()
    aux_tensor_id: str | None = None
    scalar_param: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _EVG_ARITY or not self.node_id or not self.dtype or not self.layout:
            raise ModelingError(f"evg_node_invalid:{self.node_id}:{self.kind}")
        if type(self.inputs) is not tuple or len(self.inputs) != _EVG_ARITY[self.kind]:
            raise ModelingError(f"evg_node_arity_invalid:{self.node_id}")
        if self.kind in _EVG_TENSOR_NODES and not self.aux_tensor_id:
            raise ModelingError(f"evg_aux_tensor_required:{self.node_id}")
        if self.kind in {"muls", "leaky_relu"} and (
            self.scalar_param is None or not isfinite(self.scalar_param)
        ):
            raise ModelingError(f"evg_scalar_required:{self.node_id}")


def evg_order(nodes: tuple[EvgVisitorNode, ...]) -> tuple[EvgVisitorNode, ...]:
    by_id = {node.node_id: node for node in nodes}
    if len(by_id) != len(nodes) or any(dep not in by_id for node in nodes for dep in node.inputs):
        raise ModelingError("evg_dag_unknown_or_duplicate_node")
    try:
        order = tuple(TopologicalSorter({node.node_id: node.inputs for node in nodes}).static_order())
    except CycleError as exc:
        raise ModelingError("evg_dag_cycle") from exc
    return tuple(by_id[node_id] for node_id in order)


@dataclass(frozen=True, slots=True)
class CatlassKernelSpec:
    template_id: str
    arch_tag: Literal["AtlasA2", "Ascend950"]
    dispatch_policy: str
    dtype_a: str
    dtype_b: str
    dtype_c: str
    dtype_acc: str
    layout_a: str
    layout_b: str
    layout_c: str
    prologue_a: str | None
    prologue_b: str | None
    mx_scale_group: int
    scale_layout: str | None
    l1_scale_factor_k: int
    l1_tile_shape: tuple[int, int, int]
    l0_tile_shape: tuple[int, int, int]
    stages: tuple[int, int, int, int, int]
    has_bias: bool
    enable_unit_flag: bool
    enable_l1_resident: bool
    scheduler: Literal["dataparallel", "streamk", "multi_core_splitk", "grouped_slice_m"]
    splitk_factor: int
    swizzle_offset: int
    swizzle_direction: Literal["Zn", "Nz"]
    l2_cache_hint_b: Literal["normal", "disable"]
    epilogue_kind: Literal["none", "evg_gm", "evg_ub", "grouped_swiglu_mx_quant", "finalize_routing"]
    evg_tree: tuple[EvgVisitorNode, ...]
    workspace_bytes: int
    simt_thread_block_dim: tuple[int, int, int] | None
    aic_budget: int
    aiv_budget: int
    source_refs: tuple[TemplateSourceRef | SourceRef, ...]

    def __post_init__(self) -> None:
        if bool(self.evg_tree) != (self.epilogue_kind in {"evg_gm", "evg_ub"}):
            raise ModelingError("evg_tree_epilogue_mismatch")
        evg_order(self.evg_tree)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DslKernelSpec:
    kernel_id: str
    execution_mode: Literal["cube_only", "vector_only", "mix_cube_vector"]
    algorithm: AlgorithmKind
    template_variant: str
    tile_shape: tuple[int, ...]
    ring_depths: tuple[tuple[str, str, int], ...]
    direct_routes: tuple[tuple[str, str, str], ...]
    l2_cache_policies: tuple[tuple[str, str], ...]
    workspace_bytes_total: int
    uses_aicpu_metadata: bool
    has_eager_d2h_readback: bool
    uses_simt: bool
    aic_budget: int
    aiv_budget: int
    ub_peak_bytes: int
    l1_peak_bytes: int
    l0c_peak_bytes: int
    source_refs: tuple[TemplateSourceRef | SourceRef, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TensorRangeObligation:
    """Exact logical byte coverage, with strided rows retained as address runs."""

    die: int
    allocation_id: str
    tensor_version: int
    ranges: tuple[tuple[int, int], ...]
    initial_residency: Literal["hbm_cold", "l2_warm", "onchip_resident"]
    required_final_visibility: Literal["owner", "die", "device", "host"]
    address_runs: tuple[AddressRun, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.die) is not int or self.die < 0 or not self.allocation_id
            or type(self.ranges) is not tuple or type(self.address_runs) is not tuple
            or not (self.ranges or self.address_runs)
            or type(self.tensor_version) is not int or self.tensor_version < 0
            or self.initial_residency not in {"hbm_cold", "l2_warm", "onchip_resident"}
            or self.required_final_visibility not in {"owner", "die", "device", "host"}
        ):
            raise ModelingError("tensor_range_obligation_invalid")
        end = 0
        for interval in self.ranges:
            if type(interval) is not tuple or len(interval) != 2:
                raise ModelingError(f"tensor_range_intervals_invalid:{self.allocation_id}")
            lo, hi = interval
            if type(lo) is not int or type(hi) is not int or not end <= lo < hi:
                raise ModelingError(f"tensor_range_intervals_invalid:{self.allocation_id}")
            end = hi
        for run in self.address_runs:
            if not isinstance(run, AddressRun) or (
                type(run.base_byte) is not int or type(run.stride_bytes) is not int
                or type(run.rows) is not int or type(run.row_bytes) is not int
                or run.rows <= 0 or run.row_bytes <= 0
            ):
                raise ModelingError(f"tensor_address_run_invalid:{self.allocation_id}")


@dataclass(frozen=True, slots=True)
class EdgeRouteBinding:
    src_op: str
    producer_version: int
    view_id: str
    dst_op: str
    iter_kind: Literal["tile_pipelined", "chunk_carried", "full_barrier"]
    route_kind: Literal["l0c->ub", "l0c->l1", "ub->l1", "ub->ub", "gm_spill"]
    src_owner_axis: Literal["aic", "aiv"]
    dst_owner_axis: Literal["aic", "aiv"]
    representation: str
    layout: str
    ring_depth: int
    in_place_mutates: bool = False

    def __post_init__(self) -> None:
        if (
            not all((self.src_op, self.dst_op, self.view_id, self.representation, self.layout))
            or type(self.producer_version) is not int or self.producer_version < 0
            or type(self.ring_depth) is not int or self.ring_depth < 1
            or self.iter_kind not in {"tile_pipelined", "chunk_carried", "full_barrier"}
            or self.route_kind not in {"l0c->ub", "l0c->l1", "ub->l1", "ub->ub", "gm_spill"}
            or not {self.src_owner_axis, self.dst_owner_axis} <= {"aic", "aiv"}
        ):
            raise ModelingError("fuse_region_edge_route_invalid")


@dataclass(frozen=True, slots=True)
class TileLangLoweringProfile:
    """Resolved TileLang-Ascend silicon calibration profile for template lowering."""

    api_tier: TileLangApiTier
    uses_simt: bool
    ub_ceiling_bytes: int
    simd_ub_ceiling_bytes: int
    pipelined_stages: int
    dma_pattern: TileLangDmaPattern
    dma_attainment: float
    dispatch_mode: TileLangDispatchMode
    dispatch_us: float
    vector_mode: str
    vector_gops_per_aiv: float
    cube_epilogue: TileLangCubeEpilogue
    blockscaled: bool
    hf32: bool
    calibration_sha256: str


@dataclass(frozen=True, slots=True)
class FuseRegionSpec:
    region_id: str
    op_ids: tuple[str, ...]
    live_in_obligations: tuple[TensorRangeObligation, ...]
    live_out_obligations: tuple[TensorRangeObligation, ...]
    edge_routes: tuple[EdgeRouteBinding, ...]
    reduction_class: AlgorithmKind
    tile_solution: TileSolution
    backend: FuseRegionBackend
    tilelang_profile: TileLangLoweringProfile | None = None


@dataclass(frozen=True, slots=True)
class SemanticWorkItem:
    work_id: str
    kind: Literal["matrix_mac", "vector_alu", "vector_sfu", "vector_reduce", "sort_topk"]
    dtype: str
    logical_ops: int
    admitted_engines: tuple[Literal["cube", "aiv_simd", "aiv_simt"], ...] = ("cube",)
    admissible_schedulers: tuple[str, ...] = ("dataparallel",)
    inline_epilogue_absorbable: bool = False
    die: int = 0
    replica: int = 0
    pp_stage: int = 0

    @property
    def eligible_engines(self) -> tuple[str, ...]:
        return self.admitted_engines

    def __post_init__(self) -> None:
        if (
            not self.work_id or not self.dtype or type(self.logical_ops) is not int or self.logical_ops < 0
            or not self.admitted_engines or not set(self.admitted_engines) <= {"cube", "aiv_simd", "aiv_simt"}
            or not self.admissible_schedulers
            or type(self.inline_epilogue_absorbable) is not bool
            or any(type(index) is not int or index < 0 for index in (self.die, self.replica, self.pp_stage))
            or self.kind not in {"matrix_mac", "vector_alu", "vector_sfu", "vector_reduce", "sort_topk"}
        ):
            raise ModelingError(f"template_semantic_work_invalid:{self.work_id}")


def _indexed_fp16(problem) -> bool:
    return (not problem.has_bias and not problem.groups
            and {dtype.lower() for dtype in (problem.dtype_a, problem.dtype_b, problem.dtype_c)} <= {"fp16", "float16"}
            and problem.dtype_acc.lower() in {"fp32", "float32"})


def full_load_a_eligible(problem, datapath) -> bool:
    return (datapath.arch_id in DIRECT_MATRIX_ARCH_IDS and datapath.has_l0c_to_ub and datapath.aiv_count > 0
            and _indexed_fp16(problem)
            and problem.k <= FULL_LOAD_A_MAX_K and problem.k % FP16_ELEMENTS_PER_BLOCK == 0)


def eligible_matrix_engines(problem, datapath) -> tuple[str, ...]:
    if problem.indices is not None or problem.physical_m is not None:
        engines = ("cube",) if full_load_a_eligible(problem, datapath) else ()
        if (datapath.arch_id in DIRECT_MATRIX_ARCH_IDS and datapath.simt_ub_carveout_bytes > 0
                and _indexed_fp16(problem)):
            engines += ("aiv_simt",)
        return engines
    fp16 = {"fp16", "float16"}
    if (datapath.arch_id in DIRECT_MATRIX_ARCH_IDS and datapath.simt_ub_carveout_bytes > 0
            and not problem.has_bias and not problem.groups
            and {problem.dtype_a.lower(), problem.dtype_b.lower()} <= fp16):
        return ("cube", "aiv_simt")
    return ("cube",)


@dataclass(frozen=True, slots=True)
class FusedRegionSummary:
    tier: Literal["K1", "K2", "K3"]
    region_id: str
    op_ids: tuple[str, ...]
    semantic_work: tuple[SemanticWorkItem, ...]
    issued_cube_flops_by_dtype: tuple[tuple[str, int], ...]
    vector_ops_by_pipe: tuple[tuple[str, float], ...]
    read_obligations: tuple[TensorRangeObligation, ...]
    write_obligations: tuple[TensorRangeObligation, ...]
    logical_transfer_bytes: int
    issued_copy_bytes: int
    dram_sector_bytes: int
    turnaround_service_bytes: Fraction
    realized_l2_hit_bytes: int
    comm_wire_bytes_by_link: tuple[tuple[str, int], ...]
    stars_launch_count: int
    aicpu_metadata_ms: Fraction
    sram_peaks_bytes: tuple[tuple[str, int], ...]
    spec: CatlassKernelSpec | DslKernelSpec | FuseRegionSpec | None
    # A live program publishes the latest completed invocation's directional delivery and bus service separately.
    realized_hbm_delivered_bytes: float | None = field(default=None, compare=False)
    realized_l2_miss_bytes: int = field(default=0, compare=False)
    dram_bus_service_bytes: float | None = field(default=None, compare=False)

    @property
    def realized_hbm_bytes(self) -> float | None:
        return self.realized_hbm_delivered_bytes

    @property
    def l2_hit_bytes(self) -> int:
        return self.realized_l2_hit_bytes

    @property
    def l2_miss_bytes(self) -> int:
        return self.realized_l2_miss_bytes

    def _record_delivery(self, delivered, reads, writes):
        object.__setattr__(self, "read_obligations", reads)
        object.__setattr__(self, "write_obligations", writes)
        for name in (
            "logical_transfer_bytes", "issued_copy_bytes", "dram_sector_bytes", "turnaround_service_bytes",
            "realized_l2_hit_bytes", "realized_l2_miss_bytes",
            "realized_hbm_delivered_bytes", "dram_bus_service_bytes",
        ):
            if name in delivered:
                object.__setattr__(self, name, delivered[name])


@dataclass(frozen=True, slots=True)
class TemplateLoweringResult:
    program: KernelProgram
    summary: FusedRegionSummary
    stages_count: int

    @property
    def l2_hit_bytes(self) -> int:
        return self.summary.l2_hit_bytes

    @property
    def dram_miss_bytes(self) -> int:
        return self.summary.l2_miss_bytes


def _strict_int(value: object, code: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


@dataclass(frozen=True, slots=True)
class UbAdmissionReceipt:
    """Simultaneous single-counted UB/L1/L0C/L2 admission receipt for one tier trial."""

    region_id: str
    soc_arch: SocArch
    tier_kind: LoweringTierKind
    algorithm: str
    peak_ub_bytes: int
    ub_wall_bytes: int
    admitted: bool
    working_ub_at_peak_bytes: int = 0
    ring_ub_at_peak_bytes: int = 0
    l1_peak_bytes: int = 0
    l0c_peak_bytes: int = 0
    l2_reserved_bytes: int = 0
    ub_wall_warning: str = ""
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str) or not self.region_id.strip():
            raise ModelingError("ub_admission_region_id_required")
        if self.soc_arch not in SOC_ARCHS:
            raise ModelingError(f"ub_admission_soc_arch_invalid:{self.soc_arch!r}")
        if self.tier_kind not in LOWERING_TIER_KINDS:
            raise ModelingError(f"ub_admission_tier_kind_invalid:{self.tier_kind!r}")
        if not isinstance(self.algorithm, str) or not self.algorithm.strip():
            raise ModelingError("ub_admission_algorithm_required")
        peak = _strict_int(self.peak_ub_bytes, "ub_admission_peak_ub_invalid", minimum=0)
        wall = _strict_int(self.ub_wall_bytes, "ub_admission_ub_wall_invalid", minimum=1)
        working = _strict_int(self.working_ub_at_peak_bytes, "ub_admission_working_ub_invalid", minimum=0)
        ring = _strict_int(self.ring_ub_at_peak_bytes, "ub_admission_ring_ub_invalid", minimum=0)
        _strict_int(self.l1_peak_bytes, "ub_admission_l1_invalid", minimum=0)
        _strict_int(self.l0c_peak_bytes, "ub_admission_l0c_invalid", minimum=0)
        _strict_int(self.l2_reserved_bytes, "ub_admission_l2_invalid", minimum=0)
        if not isinstance(self.admitted, bool):
            raise ModelingError("ub_admission_admitted_flag_invalid")
        if (working > 0 or ring > 0) and peak != working + ring:
            raise ModelingError(
                f"ub_admission_peak_must_equal_working_plus_ring:peak={peak}!=working({working})+ring({ring})"
            )
        if self.admitted and peak > wall:
            raise ModelingError(f"ub_admission_admitted_exceeds_ub_wall:peak={peak}>wall={wall}")
        if not self.admitted and (not isinstance(self.rejection_reason, str) or not self.rejection_reason.strip()):
            raise ModelingError("ub_admission_rejection_reason_required")

    @property
    def headroom_bytes(self) -> int:
        return self.ub_wall_bytes - self.peak_ub_bytes

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TierLoweringOutcome:
    """Independent per-tier lowering and full-step Engine evaluation outcome."""

    tier_kind: LoweringTierKind
    backend_target: str
    admitted: bool
    step_ns: int | None
    lowering_id: str = ""
    lowering_sha256: str = ""
    ub_receipt: UbAdmissionReceipt | None = None
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        if self.tier_kind not in LOWERING_TIER_KINDS:
            raise ModelingError(f"tier_lowering_outcome_tier_invalid:{self.tier_kind!r}")
        if not isinstance(self.backend_target, str) or not self.backend_target.strip():
            raise ModelingError("tier_lowering_outcome_backend_required")
        if not isinstance(self.admitted, bool):
            raise ModelingError("tier_lowering_outcome_admitted_invalid")
        if self.ub_receipt is not None:
            if not isinstance(self.ub_receipt, UbAdmissionReceipt):
                raise ModelingError("tier_lowering_outcome_ub_receipt_invalid")
            if self.ub_receipt.tier_kind != self.tier_kind:
                raise ModelingError(
                    f"tier_lowering_outcome_ub_tier_mismatch:{self.ub_receipt.tier_kind}!={self.tier_kind}"
                )
        if self.admitted:
            _strict_int(self.step_ns, "tier_lowering_outcome_step_ns_required", minimum=0)
            if self.ub_receipt is not None and not self.ub_receipt.admitted:
                raise ModelingError("tier_lowering_outcome_admitted_with_rejected_ub")
        else:
            if self.step_ns is not None:
                _strict_int(self.step_ns, "tier_lowering_outcome_step_ns_invalid", minimum=0)
            if not isinstance(self.rejection_reason, str) or not self.rejection_reason.strip():
                raise ModelingError("tier_lowering_outcome_rejection_reason_required")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DualTierLoweringReceipt:
    """Combined receipt across independent shipping/synthesis tier trials selecting strict min ns."""

    region_id: str
    predecessor_step_ns: int
    tier_outcomes: tuple[TierLoweringOutcome, ...]
    best_feasible_tier: LoweringTierKind | None = None
    best_lowering_id: str = ""
    best_lowering_sha256: str = ""
    best_step_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str) or not self.region_id.strip():
            raise ModelingError("dual_tier_receipt_region_id_required")
        _strict_int(self.predecessor_step_ns, "dual_tier_receipt_predecessor_ns_invalid", minimum=0)
        outcomes = tuple(self.tier_outcomes)
        if not outcomes or not all(isinstance(o, TierLoweringOutcome) for o in outcomes):
            raise ModelingError("dual_tier_receipt_outcomes_invalid")
        if len({o.tier_kind for o in outcomes}) != len(outcomes):
            raise ModelingError("dual_tier_receipt_duplicate_tier_kind")
        for outcome in outcomes:
            if outcome.ub_receipt is not None and outcome.ub_receipt.region_id != self.region_id:
                raise ModelingError(
                    f"dual_tier_receipt_ub_region_mismatch:{outcome.ub_receipt.region_id}!={self.region_id}"
                )
        object.__setattr__(self, "tier_outcomes", outcomes)
        admitted = [o for o in outcomes if o.admitted]
        if not admitted:
            if (
                self.best_feasible_tier is not None
                or self.best_step_ns is not None
                or bool(self.best_lowering_id)
                or bool(self.best_lowering_sha256)
            ):
                raise ModelingError("dual_tier_receipt_best_forbidden_when_none_admitted")
            return
        min_ns = min(int(o.step_ns) for o in admitted if o.step_ns is not None)
        if self.best_feasible_tier is None:
            best = min(admitted, key=lambda item: int(item.step_ns if item.step_ns is not None else 0))
            object.__setattr__(self, "best_feasible_tier", best.tier_kind)
            object.__setattr__(self, "best_lowering_id", best.lowering_id)
            object.__setattr__(self, "best_lowering_sha256", best.lowering_sha256)
            object.__setattr__(self, "best_step_ns", best.step_ns)
            return
        matched = next((o for o in admitted if o.tier_kind == self.best_feasible_tier), None)
        if matched is None:
            raise ModelingError(f"dual_tier_receipt_selected_tier_not_admitted:{self.best_feasible_tier}")
        best_ns = _strict_int(self.best_step_ns, "dual_tier_receipt_best_step_ns_required", minimum=0)
        if best_ns != matched.step_ns or best_ns > min_ns:
            raise ModelingError(
                f"dual_tier_receipt_best_step_ns_not_minimal:selected={best_ns}:matched={matched.step_ns}:min={min_ns}"
            )
        if self.best_lowering_id and self.best_lowering_id != matched.lowering_id:
            raise ModelingError("dual_tier_receipt_lowering_id_mismatch")
        if self.best_lowering_sha256 and self.best_lowering_sha256 != matched.lowering_sha256:
            raise ModelingError("dual_tier_receipt_lowering_sha256_mismatch")

    def to_dict(self) -> dict:
        return asdict(self)


def _obligation_runs(entry):
    yield from (_range_run(lo, hi) for lo, hi in entry.ranges)
    yield from entry.address_runs


def _entry_with_runs(entry, runs, **updates):
    ranges, address_runs = [], []
    for run in _coalesce_runs(runs):
        if run.rows == 1 or run.stride_bytes <= run.row_bytes:
            ranges.append((run.base_byte, _run_end(run)))
        else:
            address_runs.append(run)
    return replace(entry, ranges=_merge_byte_intervals(ranges), address_runs=tuple(address_runs), **updates)


def _logical_obligation(pattern):
    ranges, runs = [], []
    for run in pattern.runs.runs:
        if not run.rows or not run.row_bytes:
            continue
        if run.rows == 1 or run.stride_bytes <= run.row_bytes:
            ranges.append((run.base_byte, _run_end(run)))
        else:
            runs.append(run)
    if ranges or runs:
        return TensorRangeObligation(
            pattern.die, pattern.tensor_id, pattern.version, _merge_byte_intervals(ranges),
            "hbm_cold", pattern.visibility.scope, tuple(runs),
        )


def _split_entry_residency(entry, resident: SectorCoverage):
    if resident.empty:
        yield entry
        return
    data, masks = tuple(_obligation_runs(entry)), tuple(_coverage_runs(resident))
    groups = defaultdict(list)
    for state, run in _residency_runs(data, masks):
        groups[state].append(run)
    for warm, runs in groups.items():
        yield _entry_with_runs(entry, runs, initial_residency="l2_warm" if warm else entry.initial_residency)


def _union_obligations(
    obligations: Iterable[TensorRangeObligation],
) -> tuple[TensorRangeObligation, ...]:
    grouped = defaultdict(list)
    for obligation in obligations:
        grouped[obligation.die, obligation.allocation_id, obligation.tensor_version].append(obligation)
    visibility = ("owner", "die", "device", "host")
    residency = ("hbm_cold", "l2_warm", "onchip_resident")

    def classify(labels):
        return min(r for r, _ in labels), max(v for _, v in labels)

    result = []
    for _, entries in sorted(grouped.items()):
        if all((e.initial_residency, e.required_final_visibility) ==
               (entries[0].initial_residency, entries[0].required_final_visibility) for e in entries):
            runs = tuple(dict.fromkeys(run for entry in entries for run in _obligation_runs(entry)))
            periodic = tuple(run for run in runs if run.rows > 1 and run.stride_bytes > run.row_bytes)
            period = lcm(*(run.stride_bytes for run in periodic))
            if len(runs) == 1 or any(min(run.rows, period // run.stride_bytes) > _MAX_EXPLICIT_SECTOR_INTERVALS
                                      for run in periodic):
                # A union with a single label can retain overlapping runs as
                # a set of descriptors, even when their common period is large.
                result.append(_entry_with_runs(entries[0], runs))
                continue
        tagged = (
            (run, (residency.index(entry.initial_residency), visibility.index(entry.required_final_visibility)))
            for entry in entries for run in _obligation_runs(entry)
        )
        merged = defaultdict(list)
        for state, run in _partition_runs(tagged, classify):
            merged[state].append(run)
        result.extend(_entry_with_runs(entries[0], runs, initial_residency=residency[r],
                                       required_final_visibility=visibility[v]) for (r, v), runs in merged.items())
    return tuple(result)


def compose_step_obligations(
    summaries: Iterable[FusedRegionSummary],
) -> tuple[tuple[TensorRangeObligation, ...], tuple[TensorRangeObligation, ...]]:
    """Return canonical compulsory step reads and live-out writes, respectively."""
    produced, reads, writes = {}, [], []
    for summary in summaries:
        reads.extend(entry for entry in _external_reads(summary.read_obligations, produced)
                     if entry.initial_residency == "hbm_cold")
        _record_produced(produced, summary.write_obligations)
        writes.extend(entry for entry in summary.write_obligations if entry.required_final_visibility in {"device", "host"})
    return _union_obligations(reads), _union_obligations(writes)


def _external_reads(reads, produced):
    for entry in reads:
        key = entry.die, entry.allocation_id, entry.tensor_version
        if not (written := produced.get(key)):
            yield entry
            continue
        tagged = [(run, True) for run in _obligation_runs(entry)]
        tagged.extend((run, False) for run in written)
        runs = tuple(run for _, run in _partition_runs(
            tagged, lambda labels: True if True in labels and False not in labels else None,
        ))
        if runs:
            yield _entry_with_runs(entry, runs)


def _record_produced(produced, writes):
    for entry in writes:
        key = entry.die, entry.allocation_id, entry.tensor_version
        produced.setdefault(key, set()).update(_obligation_runs(entry))


def _topological(parents, message):
    try:
        return tuple(TopologicalSorter(parents).static_order())
    except CycleError as exc:
        raise ModelingError(message) from exc


def _region_order(op_ids, dependencies, barriers, edge_routes, waits, numerics, views):
    selected = set(op_ids)
    if not selected or len(selected) != len(op_ids) or not selected <= dependencies.keys():
        raise ModelingError("fuse_region_op_ids_invalid")
    if selected.intersection(barriers):
        raise ModelingError("fuse_region_collective_barrier")
    if any(parent not in dependencies for parents in dependencies.values() for parent in parents):
        raise ModelingError("fuse_region_graph_parent_unknown")
    order = _topological(dependencies, "fuse_region_graph_cycle")
    escaped = set()
    for name in order:
        if name in selected:
            if escaped.intersection(dependencies[name]):
                raise ModelingError(f"fuse_region_not_convex:{name}")
        elif (selected | escaped).intersection(dependencies[name]):
            escaped.add(name)
    internal = {name: set(dependencies[name]) & selected for name in selected}
    edge_keys = set()
    for edge in edge_routes:
        if edge.dst_op not in selected or edge.src_op not in internal[edge.dst_op]:
            raise ModelingError(f"fuse_region_edge_not_internal:{edge.view_id}")
        key = (edge.src_op, edge.producer_version, edge.view_id, edge.dst_op, edge.iter_kind)
        if key in edge_keys:
            raise ModelingError(f"fuse_region_duplicate_edge:{edge.view_id}")
        edge_keys.add(key)
    if {(edge.src_op, edge.dst_op) for edge in edge_routes} != {
        (producer, consumer) for consumer, producers in internal.items() for producer in producers
    }:
        raise ModelingError("fuse_region_internal_route_missing")
    for publisher, consumer in waits:
        if publisher not in selected or consumer not in selected:
            raise ModelingError("fuse_region_wait_endpoint_outside")
        internal[consumer].add(publisher)
    _topological(internal, "fuse_region_credit_publish_deadlock")
    if not numerics.keys() <= selected or any(before != after for before, after in numerics.values()):
        raise ModelingError("fuse_region_numerical_equivalence_failed")

    def precedes(earlier, later):
        pending = list(internal[later])
        visited = set()
        while pending:
            predecessor = pending.pop()
            if predecessor == earlier:
                return True
            if predecessor not in visited:
                visited.add(predecessor)
                pending.extend(internal[predecessor])
        return False

    for mutation in (edge for edge in edge_routes if edge.in_place_mutates):
        for reader in edge_routes:
            if reader is mutation:
                continue
            mut_view, read_view = views.get(mutation.view_id), views.get(reader.view_id)
            if mut_view is not None and read_view is not None:
                m_allocation, m_lo, m_hi = mut_view
                r_allocation, r_lo, r_hi = read_view
                aliases = m_allocation == r_allocation and max(m_lo, r_lo) < min(m_hi, r_hi)
            else:
                aliases = mutation.view_id == reader.view_id or mutation.src_op == reader.src_op
            if (aliases and reader.producer_version <= mutation.producer_version
                    and not precedes(reader.dst_op, mutation.dst_op)):
                raise ModelingError(f"fuse_region_in_place_unread_alias:{mutation.view_id}:{reader.dst_op}")
    carried = {(edge.src_op, edge.dst_op) for edge in edge_routes if edge.iter_kind == "chunk_carried"}
    same_tile = {(edge.src_op, edge.dst_op) for edge in edge_routes if edge.iter_kind != "chunk_carried"} | set(waits)
    for src, dst in carried - same_tile:
        internal[dst].discard(src)
    return tuple(name for name in order if name in selected), internal


# ---------------------------------------------------------------------------
# Layer-1 Formula Kernel & Synthesis Contracts (Single Source of Truth)
# ---------------------------------------------------------------------------

ABI_DIGEST_LEN = 12
KIB_BYTES = 1024
MXFP_BLOCK_GROUP = 32
MATMUL_SHAPE_RANK = 3
BindingState = Literal["bound_implementation", "unbound_formula_projection"]

_FORMULA_DTYPE_BYTES: Mapping[str, float] = {
    "fp8_e4m3fn": 1.0,
    "fp8_e4m3": 1.0,
    "fp8_e5m2": 1.0,
    "hif8": 1.0,
    "fp8_e8m0": 1.0,
    "fp4_e2m1": 0.5,
    "bfloat16": 2.0,
    "float16": 2.0,
    "float32": 4.0,
    "float64": 8.0,
}
_EQUATION_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_ROW_REDUCTION_FAMILIES = frozenset({
    "rmsnorm", "layernorm", "add_rmsnorm", "add_rmsnorm_quant", "rmsnorm_quant",
    "norm_gate", "l2norm", "softmax", "quantize", "dynamic_quant", "dynamic_mx_quant",
    "attention", "paged_attention", "mla_attention", "flash_attention", "attn_res",
})
_COLLECTIVE_FAMILIES = frozenset({
    "allreduce", "allgather", "reduce_scatter", "all_to_all", "alltoall",
    "moe_dispatch", "moe_combine", "all_gather_matmul", "matmul_reduce_scatter",
})
_MATMUL_FAMILIES = frozenset({
    "gemm", "matmul", "linear", "grouped_matmul", "grouped_gemm",
    "all_gather_matmul", "matmul_reduce_scatter", "quant_matmul",
})


def formula_elem_bytes(dtype: str) -> float:
    """Return exact element byte width for ``dtype``, failing fast if unknown."""
    tok = str(dtype or "").strip().lower()
    if not tok:
        raise ModelingError("formula_dtype_missing")
    if tok in _FORMULA_DTYPE_BYTES:
        return _FORMULA_DTYPE_BYTES[tok]
    return float(dtype_bytes(tok))


def _strip_op_prefix(op_id: str) -> str:
    tail = str(op_id or "").split(".")[-1]
    return re.sub(r"[^A-Za-z0-9_]+", "_", tail).strip("_")


def _extract_op_operand_ports(op: OpSpec) -> list[tuple[str, str, int]]:
    """Extract ordered ``(producer_id, port_name, version)`` activation operands preserving multiplicity."""
    ports: list[tuple[str, str, int]] = []
    meta = dict(op.metadata) if op.metadata else {}
    for key in ("typed_inputs", "input_ports"):
        raw = meta.get(key, ())
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            for entry in raw:
                if not isinstance(entry, Mapping):
                    continue
                if str(entry.get("role") or "").strip().lower() in {"weight", "control", "barrier", "event"}:
                    continue
                prod_id = entry.get("producer")
                if isinstance(prod_id, str) and prod_id and prod_id != op.op_id:
                    pname = str(entry.get("port") or entry.get("output") or entry.get("name") or "output")
                    ver = int(entry.get("version", entry.get("state_version", 0)) or 0)
                    ports.append((prod_id, "output" if pname == "out" else pname, ver))
        if ports:
            break
    if not ports:
        for dep, port in input_port_entries(op.metadata):
            if dep and dep != op.op_id:
                pname = port.removeprefix(dep + ":")
                if pname != "weight":
                    ports.append((dep, "output" if pname == "out" else pname, 0))
    explicit_producers = {p[0] for p in ports}
    for dep in op.deps:
        if dep and dep != op.op_id and dep not in explicit_producers:
            ports.append((dep, "output", 0))
    return ports


def _infer_op_symbolic_kind(op: OpSpec) -> tuple[str, bool, bool, str, tuple[tuple[str, float], ...]]:
    """Classify ``op`` locally as ``(kind, is_matrix, is_row_red, activation, act_params)``."""
    fam = str(op.family or "").strip().lower()
    rec = str(op.recipe or "").strip().lower()
    meta = dict(op.metadata) if op.metadata else {}
    act = str(meta.get("activation") or "").strip().lower()
    act_params_raw = meta.get("activation_params") or ()
    if not act and isinstance(act_params_raw, Mapping):
        act = str(act_params_raw.get("activation") or "").strip().lower()
    act_params_iter = act_params_raw.items() if isinstance(act_params_raw, Mapping) else act_params_raw
    act_params = tuple(
        (str(k), float(v))
        for k, v in act_params_iter
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    )
    has_flops = bool(op.matrix_flops) or float(op.flops or 0.0) > 0.0
    is_matrix = has_flops or fam in _MATMUL_FAMILIES
    is_row_red = (
        fam in _ROW_REDUCTION_FAMILIES
        or any(t in fam or t in rec for t in ("rmsnorm", "layernorm", "softmax", "quant", "attn_res"))
    )
    if "megamoe" in fam or "megamoe" in rec:
        return "megamoe", True, True, act or "swiglu", act_params
    if "gating" in fam and "topk" in rec:
        return "moe_router_topk", False, True, act, act_params
    if fam in {"all_gather_matmul", "matmul_reduce_scatter", "allgather", "reduce_scatter"}:
        return fam, fam in {"all_gather_matmul", "matmul_reduce_scatter"}, False, act, act_params
    if fam in {"grouped_matmul", "grouped_gemm"}:
        return "grouped_matmul", True, False, act, act_params
    if is_matrix and fam in _MATMUL_FAMILIES:
        return "matmul", True, False, act, act_params
    if fam in {"view", "reshape", "split", "splitv", "slice", "transpose"}:
        return "view_copy", False, False, act, act_params
    if "situ" in fam or "situ" in rec or act == "situ":
        return "situ", False, False, "situ", act_params
    if "swiglu" in fam or "swiglu" in rec or act == "swiglu":
        return "swiglu", False, False, "swiglu", act_params
    if "add_rmsnorm" in fam or "add_rmsnorm" in rec:
        k = "add_rmsnorm_quant" if "quant" in fam or "quant" in rec else "add_rmsnorm"
        return k, False, True, act, act_params
    if "rmsnorm" in fam or "rmsnorm" in rec:
        k = "rmsnorm_quant" if "quant" in fam or "quant" in rec else "rmsnorm"
        return k, False, True, act, act_params
    if "mx" in fam and "quant" in fam:
        return "dynamic_mx_quant", False, True, act, act_params
    if "quant" in fam:
        return "dynamic_quant", False, True, act, act_params
    return fam or "op", is_matrix, is_row_red, act, act_params


def _format_megamoe_equation(
    prev: str,
    out: str,
    shape_tag: str,
    param_txt: str,
    act_kind: str,
    meta: Mapping[str, Any],
) -> str:
    ep = int(meta.get("ep", meta.get("ep_size", 1)) or 1)
    has_ep = ep > 1 or bool(meta.get("all_to_all") or meta.get("ep_dispatch"))
    comp_dt = str(meta.get("compute_dtype") or meta.get("weight_format") or meta.get("op_dtype") or "").strip()
    act = act_kind or str(meta.get("activation") or "swiglu")
    has_shared = bool(meta.get("has_shared_expert") or int(meta.get("shared_expert_intermediate", 0) or 0) > 0)
    shared_inter = int(meta.get("shared_expert_intermediate", 0) or 0)
    disp_step = f"h_disp = AllToAllDispatch({prev}, ep={ep})" if has_ep else f"h_disp = LocalTokenPermute({prev})"
    dt_tag = f" [{comp_dt}]" if comp_dt else ""
    gmm1 = f"u = GroupedMatMul(h_disp, W_gate_up^T{dt_tag})"
    act_step = f"a, s = ActMxQuant({act}(u{param_txt}))"
    gmm2 = "y_loc = GroupedMatMul(a, W_down^T, scales=s)"
    comb_expr = "AllToAllCombine(y_loc, topk_w)" if has_ep else "LocalWeightedCombine(y_loc, topk_w)"
    if has_shared:
        shared_tag = f", inter={shared_inter}" if shared_inter > 0 else ""
        tail = f"y_routed = {comb_expr}; y_shared = SharedFFN({prev}{shared_tag}); {out} = y_routed + y_shared"
    else:
        tail = f"{out} = {comb_expr}"
    return f"{disp_step}; {gmm1}; {act_step}; {gmm2}; {tail} [{shape_tag}, Persistent Megakernel]"


def _format_moe_gating_equation(
    prev: str,
    out: str,
    shape_tag: str,
    meta: Mapping[str, Any],
) -> str:
    raw_mode = str(meta.get("router_mode") or meta.get("score_function") or "").strip().lower()
    score_fn = raw_mode if raw_mode in {"softmax", "sigmoid"} else "softmax/sigmoid"
    noaux = bool(meta.get("noaux_tc"))
    groups = int(meta.get("groups", 0) or 0)
    group_topk = int(meta.get("group_topk", 0) or 0)
    top_k = int(meta.get("top_k", 0) or 0)
    has_bias = bool(meta.get("correction_bias") or meta.get("bias") or noaux)
    norm_topk = meta.get("norm_topk")
    scale = meta.get("routed_scaling_factor")
    sel_name = "group_limited_topk" if (noaux or groups > 1) else "TopK"
    score_expr = f"{score_fn}({prev})"
    if has_bias:
        score_expr = f"{score_expr} + e_bias"
    extras: list[str] = []
    if top_k > 0:
        extras.append(f"k={top_k}")
    if groups > 1:
        extras.append(f"groups={groups}")
    if group_topk > 0:
        extras.append(f"group_topk={group_topk}")
    if norm_topk is not None:
        extras.append(f"norm={bool(norm_topk)}")
    if scale is not None and float(scale) != 1.0:
        extras.append(f"scale={float(scale):g}")
    extra_txt = (", " + ", ".join(extras)) if extras else ""
    return f"{out}, topk_ids = {sel_name}({score_expr}{extra_txt}) [{shape_tag}, UB FP32]"


def _format_dag_equation(
    op_id: str,
    shape: Sequence[int],
    kind: str,
    input_syms: Sequence[str],
    out_sym: str,
    *,
    output_dtype: str = "bf16",
    activation_kind: str = "",
    activation_params: Sequence[tuple[str, float]] = (),
    group_size: int = 0,
    collective_axis: str = "",
    op_meta: Mapping[str, Any] | None = None,
) -> str:
    prev = input_syms[0] if input_syms else "x"
    rhs_second = input_syms[1] if len(input_syms) > 1 else ""
    all_args = ", ".join(input_syms) if input_syms else "x"
    out = out_sym
    meta = op_meta or {}
    gs = group_size or MXFP_BLOCK_GROUP
    shape_tag = "x".join(str(d) for d in shape) if shape else "?"
    param_txt = (
        "; " + ", ".join(f"{pk}={pv:g}" for pk, pv in activation_params)
        if activation_params else ""
    )
    w_tag = _strip_op_prefix(op_id)
    w_sym = f"W_{w_tag}" if w_tag else "W"
    if kind == "megamoe":
        return _format_megamoe_equation(prev, out, shape_tag, param_txt, activation_kind, meta)
    if kind == "moe_router_topk":
        return _format_moe_gating_equation(prev, out, shape_tag, meta)
    eq_map = {
        "view_copy": f"{out} = strided_view({prev}) [zero-copy metadata elision]",
        "grouped_matmul": f"{out} = GroupedMatMul({prev}, {w_sym}_e^T) [{shape_tag}, L0C FP32]",
        "matmul": f"{out} = {prev} @ {w_sym}^T [{shape_tag}, L0C FP32]",
        "allgather": f"{out} = AllGather({prev}, axis={collective_axis or 'tp'})",
        "reduce_scatter": f"{out} = ReduceScatter({prev}, SUM, axis={collective_axis or 'tp'})",
        "all_gather_matmul": f"{out} = AllGather({prev}) @ {w_sym}_col^T [MC2 pipelined]",
        "matmul_reduce_scatter": f"{out} = ReduceScatter({prev} @ {w_sym}_row^T, SUM) [MC2 pipelined]",
        "add": f"{out} = {prev} + {rhs_second or 'r'} [UB FP32]",
        "mul": f"{out} = {prev} * {rhs_second or 'g'} [UB FP32]",
        "rmsnorm": f"{out} = ({prev} / sqrt(mean({prev}^2) + eps)) * gamma [UB FP32]",
        "layernorm": f"{out} = (({prev} - mu) / sqrt(var + eps)) * gamma + beta [UB FP32]",
        "add_rmsnorm": f"{out} = RMSNorm({prev} + {rhs_second or 'r'}; gamma) [UB FP32]",
        "add_rmsnorm_quant": f"{out}, s = DynamicQuant(RMSNorm({prev} + {rhs_second or 'r'}; gamma)) [UB FP32]",
        "rmsnorm_quant": f"{out}, s = DynamicQuant(RMSNorm({prev}; gamma)) [UB FP32]",
        "norm_gate": f"{out} = RMSNorm({prev}; gamma) * SiLU({rhs_second or 'z'}) [UB FP32]",
        "l2norm": f"{out} = {prev} / (||{prev}||_2 + eps) [UB FP32]",
        "swiglu": f"{out} = SiLU({prev}_gate) * {prev}_up [UB FP32]",
        "situ": f"{out} = SiTU({prev}_gate, {prev}_up{param_txt}) [UB FP32, custom ABI]",
        "silu": f"{out} = {prev} * sigmoid({prev}) [UB FP32]",
        "gelu": f"{out} = GELU({prev}) [UB FP32]",
        "sigmoid": f"{out} = sigmoid({prev}) [UB FP32]",
        "dynamic_mx_quant": (
            f"s_b = 2^floor(log2(max|{prev}_{{b:b+{gs}}}|/448)); "
            f"{out} = cast_fp8({prev}/s_b) [E8M0 g={gs}, UB FP32]"
        ),
        "dynamic_quant": f"s = max|{prev}| / qmax; {out} = cast_{output_dtype}({prev}/s) [UB FP32]",
        "gating": f"{out} = -exp(A_log) * softplus({prev} + dt_bias) [UB FP32]",
        "conv1d": f"{out}, S_next = CausalConv1d({prev}, S_conv, W_conv) [rolling UB]",
        "recurrent_scan": f"{out}, H_next = RecurrentStateUpdate({prev}, H_prev) [FP32 state]",
        "recover_ssm": "H_commit = AliasSlotSelect(H_slots, accept_idx) [2D slot alias]",
        "rotary": f"{out} = RoPE({all_args}, cos, sin) [UB FP32]",
        "kv_scatter": f"KV_cache[slot] = QuantScatter({prev}) [in-place cache write]",
        "attn_res": f"{out} = Sum_i softmax_i(Norm(h_i) . q_l) * h_i [UB FP32]",
        "mhc_mix": f"{out} = SinkhornMix({prev}, H_pre, H_post) [UB FP32]",
        "attention_update": f"{out}, lse = OnlineSoftmaxMerge({prev}, lse_prev) [FP32]",
        "cast": f"{out} = cast_{output_dtype}({prev})",
    }
    return eq_map.get(kind, f"{out} = {kind}({all_args}{param_txt}) [{shape_tag}]")


@dataclass(frozen=True, slots=True)
class FormulaTensorSpec:
    """Declarative tensor operand for Layer-1 formula-driven modeling."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    role: Literal["input", "output", "weight", "workspace"] = "input"
    hbm_resident: bool = True
    port: str = ""
    version: int = 0

    @property
    def num_elements(self) -> int:
        if not self.shape:
            return 0
        dims = tuple(int(d) for d in self.shape)
        if any(d <= 0 for d in dims):
            return 0
        return prod(dims)

    @property
    def byte_size(self) -> float:
        return float(self.num_elements) * formula_elem_bytes(self.dtype)


@dataclass(frozen=True, slots=True)
class PipeStageSpec:
    """One causal compute/communication pipe stage derived from the true operand DAG."""

    name: str
    pipe_kind: Literal["cube", "vector", "link"]
    share: float = 1.0
    after: tuple[str, ...] = ()
    row_reduction: bool = False
    fixpipe_share: float = 0.0


def _condense_pipe_nodes(
    elem_nodes: Sequence[tuple[str, Literal["cube", "vector", "link"], float, bool, tuple[str, ...]]],
    fixpipe_by_id: Mapping[str, float] | None = None,
) -> tuple[PipeStageSpec, ...]:
    """Condense elementary pipe DAG without merging nodes whose dependency/consumer sets differ."""
    if not elem_nodes:
        return ()
    fp_map = fixpipe_by_id or {}
    node_ids = {eid for eid, _, _, _, _ in elem_nodes}
    preds_by_id: dict[str, tuple[str, ...]] = {
        eid: tuple(dict.fromkeys(p for p in preds if p in node_ids and p != eid))
        for eid, _, _, _, preds in elem_nodes
    }
    succs_by_id: dict[str, list[str]] = {eid: [] for eid in node_ids}
    for eid, plist in preds_by_id.items():
        for p in plist:
            succs_by_id[p].append(eid)

    level_by_id: dict[str, int] = {}
    visiting: set[str] = set()

    def _level(eid: str) -> int:
        if eid in level_by_id:
            return level_by_id[eid]
        if eid in visiting:
            return 0
        visiting.add(eid)
        lvl = max((_level(p) + 1 for p in preds_by_id[eid]), default=0)
        visiting.discard(eid)
        level_by_id[eid] = lvl
        return lvl

    for eid in preds_by_id:
        _level(eid)

    GroupKey = tuple[int, Literal["cube", "vector", "link"], bool, frozenset[str], frozenset[str]]
    group_key_by_id: dict[str, GroupKey] = {}
    group_first_pos: dict[GroupKey, int] = {}
    group_order: list[GroupKey] = []
    group_work: dict[GroupKey, float] = defaultdict(float)
    group_fixpipe: dict[GroupKey, float] = defaultdict(float)
    group_red: dict[GroupKey, bool] = defaultdict(bool)
    group_preds: dict[GroupKey, list[GroupKey]] = defaultdict(list)

    for pos, (eid, pkind, _work, is_red, _preds) in enumerate(elem_nodes):
        gk: GroupKey = (
            level_by_id[eid],
            pkind,
            bool(is_red) if pkind == "vector" else False,
            frozenset(preds_by_id[eid]),
            frozenset(succs_by_id[eid]),
        )
        group_key_by_id[eid] = gk
        if gk not in group_first_pos:
            group_first_pos[gk] = pos
            group_order.append(gk)

    for eid, _pkind, work, is_red, _preds in elem_nodes:
        gk = group_key_by_id[eid]
        group_work[gk] += max(0.0, float(work))
        group_fixpipe[gk] += max(0.0, float(fp_map.get(eid, 0.0)))
        if is_red:
            group_red[gk] = True
        for p in preds_by_id[eid]:
            pgk = group_key_by_id[p]
            if pgk != gk and pgk not in group_preds[gk]:
                group_preds[gk].append(pgk)

    kind_rank = {"link": 0, "vector": 1, "cube": 2}
    group_order.sort(key=lambda item: (item[0], kind_rank[item[1]], group_first_pos[item]))
    counts_by_kind: dict[str, int] = defaultdict(int)
    for gk in group_order:
        counts_by_kind[gk[1]] += 1

    base_names = {"cube": "cube_mma", "vector": "aiv_fp32", "link": "link_pipe"}
    seen_kind: dict[str, int] = defaultdict(int)
    stage_name: dict[GroupKey, str] = {}
    totals_by_kind: dict[str, float] = defaultdict(float)
    total_fixpipe = 0.0
    for gk in group_order:
        pkind = gk[1]
        idx = seen_kind[pkind]
        seen_kind[pkind] += 1
        stage_name[gk] = base_names[pkind] if counts_by_kind[pkind] == 1 else f"{base_names[pkind]}_{idx}"
        totals_by_kind[pkind] += group_work[gk]
        total_fixpipe += group_fixpipe[gk]

    stages: list[PipeStageSpec] = []
    for gk in group_order:
        pkind = gk[1]
        tot = totals_by_kind[pkind]
        share = (group_work[gk] / tot) if tot > 0.0 else (1.0 / float(counts_by_kind[pkind]))
        fp_share = (group_fixpipe[gk] / total_fixpipe) if total_fixpipe > 0.0 else 0.0
        after = tuple(stage_name[pgk] for pgk in group_preds[gk] if pgk in stage_name)
        stages.append(PipeStageSpec(
            name=stage_name[gk],
            pipe_kind=pkind,
            share=share,
            after=after,
            row_reduction=bool(group_red[gk]) if pkind == "vector" else False,
            fixpipe_share=fp_share,
        ))
    return tuple(stages)


def _derive_pipe_stages_from_ops(ops: Sequence[OpSpec]) -> tuple[PipeStageSpec, ...]:
    """Derive causal ``PipeStageSpec`` DAG strictly from true operand dependencies."""
    ordered = tuple(ops)
    if not ordered:
        return ()
    by_id = {op.op_id: op for op in ordered}
    member_ids = set(by_id)
    is_mat_by_id = {op.op_id: _infer_op_symbolic_kind(op)[1] for op in ordered}
    elem_nodes: list[tuple[str, Literal["cube", "vector", "link"], float, bool, tuple[str, ...]]] = []
    fixpipe_by_id: dict[str, float] = {}
    exit_eid: dict[str, tuple[str, ...]] = {}

    for op in ordered:
        ports = _extract_op_operand_ports(op)
        int_preds = tuple(dict.fromkeys(p[0] for p in ports if p[0] in member_ids))
        pred_eids = tuple(dict.fromkeys(eid for pid in int_preds for eid in exit_eid.get(pid, ())))
        fam = str(op.family or "").strip().lower()
        _, is_matrix, is_red, _, _ = _infer_op_symbolic_kind(op)
        cube_w = sum(float(fl) for _, fl in op.matrix_flops) if op.matrix_flops else float(op.flops or 0.0)
        vec_w = float(op.vector_ops or 0.0)
        link_w = float(op.link_bytes or 0.0)

        if fam == "all_gather_matmul":
            l_id, c_id = f"{op.op_id}:link", f"{op.op_id}:cube"
            elem_nodes.append((l_id, "link", max(link_w, 1.0), False, pred_eids))
            elem_nodes.append((c_id, "cube", max(cube_w, 1.0), False, (l_id,)))
            exit_eid[op.op_id] = (c_id,)
            continue
        if fam == "matmul_reduce_scatter":
            c_id, l_id = f"{op.op_id}:cube", f"{op.op_id}:link"
            elem_nodes.append((c_id, "cube", max(cube_w, 1.0), False, pred_eids))
            elem_nodes.append((l_id, "link", max(link_w, 1.0), False, (c_id,)))
            exit_eid[op.op_id] = (l_id,)
            continue

        local_ids: list[str] = []
        if cube_w > 0.0 or is_matrix:
            cid = f"{op.op_id}:cube"
            elem_nodes.append((cid, "cube", max(cube_w, 1.0), False, pred_eids))
            local_ids.append(cid)
        if vec_w > 0.0 or (not local_ids and fam not in _COLLECTIVE_FAMILIES and fam not in {"view", "reshape", "split", "splitv"}):
            vid = f"{op.op_id}:vec"
            v_preds = tuple(local_ids) if local_ids else pred_eids
            elem_nodes.append((vid, "vector", max(vec_w, 1.0), is_red, v_preds))
            fp_b = 0.0
            for prod_id, pname, ver in ports:
                src = by_id.get(prod_id)
                if src is not None and is_mat_by_id.get(prod_id):
                    fp_b += _op_port_edge_bytes(src, op, pname, ver, src_is_matrix=True)
            if fp_b > 0.0:
                cap_b = float(op.mte2_bytes or 0.0)
                fixpipe_by_id[vid] = min(fp_b, cap_b) if cap_b > 0.0 else fp_b
            local_ids = [vid]
        if link_w > 0.0 or (not local_ids and fam in _COLLECTIVE_FAMILIES):
            lid = f"{op.op_id}:link"
            l_preds = tuple(local_ids) if local_ids else pred_eids
            elem_nodes.append((lid, "link", max(link_w, 1.0), False, l_preds))
            local_ids = [lid]
        exit_eid[op.op_id] = tuple(local_ids) if local_ids else pred_eids
    return _condense_pipe_nodes(elem_nodes, fixpipe_by_id)


def _derive_pipe_stages_from_equations(
    equations: Sequence[str],
    *,
    has_cube: bool,
    has_vector: bool,
    has_link: bool,
    row_reduction: bool,
) -> tuple[PipeStageSpec, ...]:
    """Infer causal ``PipeStageSpec`` DAG from declarative ``FormulaKernelSpec.equations``."""
    elem_nodes: list[tuple[str, Literal["cube", "vector", "link"], float, bool, tuple[str, ...]]] = []
    producer_by_sym: dict[str, str] = {}
    for idx, eq in enumerate(equations):
        lhs, sep, rhs = str(eq).partition("=")
        rhs_txt = rhs if sep else lhs
        rhs_low = rhs_txt.lower()
        if any(tok in rhs_low for tok in ("allgather", "reducescatter", "allreduce", "alltoall")):
            pkind: Literal["cube", "vector", "link"] = "link"
        elif "@" in rhs_txt or "matmul" in rhs_low:
            pkind = "cube"
        else:
            pkind = "vector"
        if (pkind == "cube" and not has_cube) or (pkind == "vector" and not has_vector) or (pkind == "link" and not has_link):
            continue
        rhs_idents = _EQUATION_IDENT_RE.findall(rhs_txt)
        preds = tuple(dict.fromkeys(producer_by_sym[tok] for tok in rhs_idents if tok in producer_by_sym))
        eid = f"eq:{idx}"
        is_red = pkind == "vector" and (row_reduction or any(t in rhs_low for t in ("norm", "softmax", "quant")))
        elem_nodes.append((eid, pkind, 1.0, is_red, preds))
        if sep:
            for sym in _EQUATION_IDENT_RE.findall(lhs):
                producer_by_sym[sym] = eid
    return _condense_pipe_nodes(elem_nodes)


def _op_input_shape(op: OpSpec, is_matrix: bool) -> tuple[int, ...]:
    if not op.local_shape:
        return ()
    if is_matrix and len(op.local_shape) == MATMUL_SHAPE_RANK:
        return (int(op.local_shape[0]), int(op.local_shape[2]))
    return tuple(int(d) for d in op.local_shape)


def _op_output_shape(op: OpSpec, is_matrix: bool) -> tuple[int, ...]:
    if not op.local_shape:
        return ()
    if is_matrix and len(op.local_shape) == MATMUL_SHAPE_RANK:
        return (int(op.local_shape[0]), int(op.local_shape[1]))
    return tuple(int(d) for d in op.local_shape)


def _resolve_op_input_dtype(op: OpSpec, meta: Mapping[str, Any] | None = None) -> str:
    """Resolve activation input dtype honoring canonical ``TensorPrecisionABI`` first."""
    m = meta if meta is not None else (dict(op.metadata) if op.metadata else {})
    abi = getattr(op, "precision_abi", None)
    return str(
        (getattr(abi, "input_dtype", "") if abi is not None else "")
        or m.get("activation_format")
        or m.get("input_format")
        or m.get("input_dtype")
        or op.dtype
    )


def _resolve_op_output_dtype(op: OpSpec, meta: Mapping[str, Any] | None = None) -> str:
    """Resolve primary output dtype honoring canonical ``TensorPrecisionABI`` first."""
    m = meta if meta is not None else (dict(op.metadata) if op.metadata else {})
    abi = getattr(op, "precision_abi", None)
    return str(
        (getattr(abi, "output_dtype", "") if abi is not None else "")
        or m.get("output_format")
        or m.get("output_dtype")
        or m.get("payload_dtype")
        or op.dtype
    )


def _shape_from_dims_or_bytes(
    raw_shape: Any,
    raw_bytes: float,
    dtype: str,
    *,
    fallback_rows: int = 0,
    fallback_cols: int = 0,
) -> tuple[int, ...]:
    if isinstance(raw_shape, Sequence) and not isinstance(raw_shape, (str, bytes)):
        dims = tuple(int(d) for d in raw_shape if isinstance(d, int) and not isinstance(d, bool) and d > 0)
        if dims and len(dims) == len(raw_shape):
            return dims
    if raw_bytes > 0.0:
        elems = max(1, round(raw_bytes / formula_elem_bytes(dtype)))
        if fallback_rows > 0 and fallback_cols > 0 and fallback_rows * fallback_cols == elems:
            return (fallback_rows, fallback_cols)
        return (elems,)
    return ()


def _typed_input_entry(
    op: OpSpec, prod_id: str, pname: str, ver: int,
) -> Mapping[str, Any] | None:
    """Return matching ``typed_inputs`` / ``input_ports`` descriptor for an operand edge."""
    meta = dict(op.metadata) if op.metadata else {}
    for key in ("typed_inputs", "input_ports"):
        raw = meta.get(key, ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            continue
        for entry in raw:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("role") or "").strip().lower() in {"weight", "control", "barrier", "event"}:
                continue
            ep = entry.get("producer")
            if isinstance(ep, str) and ep == prod_id:
                raw_name = str(entry.get("port") or entry.get("output") or entry.get("name") or "output")
                norm_name = "output" if raw_name == "out" else raw_name
                e_ver = int(entry.get("version", entry.get("state_version", 0)) or 0)
                if norm_name == pname and e_ver == ver:
                    return entry
    return None


def _typed_output_entry(op: OpSpec, pname: str, ver: int) -> Mapping[str, Any] | None:
    """Return matching ``typed_outputs`` descriptor on producer ``op``."""
    meta = dict(op.metadata) if op.metadata else {}
    raw = meta.get("typed_outputs")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    for entry in raw:
        if not isinstance(entry, Mapping):
            continue
        raw_name = str(entry.get("port") or entry.get("output") or entry.get("name") or "output")
        norm_name = "output" if raw_name == "out" else raw_name
        e_ver = int(entry.get("version", entry.get("state_version", 0)) or 0)
        if norm_name == pname and e_ver == ver:
            return entry
    return None


def _entry_byte_size(entry: Mapping[str, Any], default_dtype: str) -> float:
    nbytes = float(entry.get("bytes", entry.get("nbytes", entry.get("byte_size", 0.0))) or 0.0)
    if nbytes > 0.0:
        return nbytes
    dt = str(entry.get("dtype") or default_dtype)
    sh = _shape_from_dims_or_bytes(entry.get("shape"), 0.0, dt)
    return float(prod(sh)) * formula_elem_bytes(dt) if sh else 0.0


def _op_port_edge_bytes(
    src: OpSpec, dst: OpSpec, pname: str, ver: int, *, src_is_matrix: bool,
) -> float:
    """Compute exact byte volume flowing across a single ``(src, pname, ver) -> dst`` edge."""
    in_entry = _typed_input_entry(dst, src.op_id, pname, ver)
    if in_entry is not None:
        size = _entry_byte_size(in_entry, _resolve_op_input_dtype(dst))
        if size > 0.0:
            return size
    out_entry = _typed_output_entry(src, pname, ver)
    if out_entry is not None:
        size = _entry_byte_size(out_entry, _resolve_op_output_dtype(src))
        if size > 0.0:
            return size
    src_meta = dict(src.metadata) if src.metadata else {}
    if "scale" in pname and float(src_meta.get("activation_scale_bytes") or 0.0) > 0.0:
        return float(src_meta["activation_scale_bytes"])
    sh = _op_output_shape(src, src_is_matrix)
    if sh:
        shape_b = float(prod(sh)) * formula_elem_bytes(_resolve_op_output_dtype(src, src_meta))
        src_b = float(src.mte3_bytes or 0.0)
        dst_b = float(dst.mte2_bytes or 0.0)
        caps = [b for b in (shape_b, src_b, dst_b) if b > 0.0]
        return min(caps) if caps else 0.0
    return min(float(src.mte3_bytes or 0.0), float(dst.mte2_bytes or 0.0))


def _op_declared_outputs(
    op: OpSpec,
    sym: str,
    is_matrix: bool,
    meta: Mapping[str, Any],
) -> list[FormulaTensorSpec]:
    """Emit all declared output port specs (typed outputs, quant scales, state writebacks)."""
    outs: list[FormulaTensorSpec] = []
    seen_ports: set[tuple[str, int]] = set()
    used_names: set[str] = set()

    def _unique_name(base: str) -> str:
        cand = base
        suffix = 1
        while cand in used_names:
            cand = f"{base}_{suffix}"
            suffix += 1
        used_names.add(cand)
        return cand

    raw_typed = meta.get("typed_outputs")
    parsed_typed: list[tuple[str, int, tuple[int, ...], str]] = []
    if isinstance(raw_typed, Sequence) and not isinstance(raw_typed, (str, bytes)):
        for entry in raw_typed:
            if not isinstance(entry, Mapping):
                continue
            pname = str(entry.get("port") or entry.get("output") or entry.get("name") or "output")
            norm_port = "output" if pname == "out" else pname
            ver = int(entry.get("version", entry.get("state_version", 0)) or 0)
            dt = str(entry.get("dtype") or _resolve_op_output_dtype(op, meta))
            nbytes = float(entry.get("bytes", entry.get("nbytes", entry.get("byte_size", 0.0))) or 0.0)
            sh = _shape_from_dims_or_bytes(entry.get("shape"), nbytes, dt)
            if not sh or (norm_port, ver) in seen_ports:
                continue
            seen_ports.add((norm_port, ver))
            parsed_typed.append((norm_port, ver, sh, dt))

    if parsed_typed:
        primary_idx = next((i for i, item in enumerate(parsed_typed) if item[0] == "output"), 0)
        for idx, (norm_port, ver, sh, dt) in enumerate(parsed_typed):
            tag = re.sub(r"[^A-Za-z0-9_]+", "_", norm_port).strip("_") or "extra"
            name = _unique_name(sym if idx == primary_idx else f"{sym}_{tag}")
            outs.append(FormulaTensorSpec(name, sh, dt, "output", port=norm_port, version=ver))
    else:
        sh = _op_output_shape(op, is_matrix)
        if sh:
            out_dt = _resolve_op_output_dtype(op, meta)
            seen_ports.add(("output", 0))
            outs.append(FormulaTensorSpec(_unique_name(sym), sh, out_dt, "output", port="output", version=0))

    scale_bytes = float(meta.get("activation_scale_bytes") or 0.0)
    scale_shape_raw = meta.get("activation_scale_shape")
    if (scale_bytes > 0.0 or scale_shape_raw) and not any("scale" in p for p, _ in seen_ports):
        abi = getattr(op, "precision_abi", None)
        scale_dt = str(
            (getattr(abi, "act_scale_dtype", "") if abi is not None else "")
            or meta.get("activation_scale_format")
            or meta.get("scale_dtype")
            or "fp32"
        )
        scale_sh = _shape_from_dims_or_bytes(scale_shape_raw, scale_bytes, scale_dt)
        if scale_sh:
            seen_ports.add(("activation_scale", 0))
            outs.append(FormulaTensorSpec(
                _unique_name(f"{sym}_scale"), scale_sh, scale_dt, "output", port="activation_scale", version=0,
            ))

    state_bytes = float(meta.get("state_write_bytes") or 0.0)
    if state_bytes > 0.0 and not any("state" in p for p, _ in seen_ports):
        contract = meta.get("gdn_state_contract")
        contract_map = contract if isinstance(contract, Mapping) else {}
        explicit_state_dt = str(
            meta.get("state_dtype")
            or contract_map.get("conv_state_dtype")
            or contract_map.get("state_dtype")
            or ""
        )
        fallback_state_dt = explicit_state_dt or str(op.dtype)
        raw_state_outs = meta.get("state_value_outputs") or contract_map.get("value_outputs") or ()
        state_entries = (
            tuple(e for e in raw_state_outs if isinstance(e, Mapping))
            if isinstance(raw_state_outs, Sequence) and not isinstance(raw_state_outs, (str, bytes))
            else ()
        )
        state_rows = int(meta.get("state_rows") or meta.get("state_history_slots") or 0)
        state_dim = int(meta.get("dim") or 0)
        if state_entries:
            resolved_state: list[tuple[int, str, int, str, tuple[int, ...] | None]] = []
            used_bytes = 0.0
            missing_indices: list[int] = []
            for idx, entry in enumerate(state_entries):
                t_port = str(entry.get("output") or entry.get("port") or entry.get("name") or f"state_{idx}")
                t_ver = int(entry.get("state_version", entry.get("version", 0)) or 0)
                t_dt = explicit_state_dt or str(entry.get("dtype") or fallback_state_dt)
                entry_b = float(entry.get("bytes", entry.get("nbytes", entry.get("byte_size", 0.0))) or 0.0)
                t_sh = _shape_from_dims_or_bytes(
                    entry.get("shape"), entry_b, t_dt,
                    fallback_rows=state_rows, fallback_cols=state_dim,
                )
                if t_sh:
                    used_bytes += float(prod(t_sh)) * formula_elem_bytes(t_dt)
                    resolved_state.append((idx, t_port, t_ver, t_dt, t_sh))
                else:
                    missing_indices.append(idx)
                    resolved_state.append((idx, t_port, t_ver, t_dt, None))
            if missing_indices:
                rem_bytes = max(0.0, state_bytes - used_bytes)
                elem_b = formula_elem_bytes(fallback_state_dt)
                total_elems = max(len(missing_indices), round(rem_bytes / elem_b)) if rem_bytes > 0.0 else 0
                base_elems, extra_elems = divmod(total_elems, len(missing_indices))
                missing_pos = {idx: pos for pos, idx in enumerate(missing_indices)}
                for pos_r, (idx, t_port, t_ver, t_dt, t_sh) in enumerate(resolved_state):
                    if t_sh is not None:
                        continue
                    elems = base_elems + (1 if missing_pos[idx] < extra_elems else 0)
                    if elems <= 0:
                        continue
                    sh_fill = (
                        (state_rows, state_dim)
                        if (state_rows > 0 and state_dim > 0 and state_rows * state_dim == elems)
                        else (elems,)
                    )
                    resolved_state[pos_r] = (idx, t_port, t_ver, t_dt, sh_fill)
            for idx, t_port, t_ver, t_dt, t_sh in resolved_state:
                if not t_sh:
                    continue
                t_name = _unique_name(f"{sym}_state" if len(state_entries) == 1 else f"{sym}_state{idx}")
                outs.append(FormulaTensorSpec(t_name, t_sh, t_dt, "output", port=t_port, version=t_ver))
        else:
            st_sh = _shape_from_dims_or_bytes(
                meta.get("state_shape"), state_bytes, fallback_state_dt,
                fallback_rows=state_rows, fallback_cols=state_dim,
            )
            if st_sh:
                outs.append(FormulaTensorSpec(
                    _unique_name(f"{sym}_state"), st_sh, fallback_state_dt, "output", port="state", version=0,
                ))
    return outs


def _compose_dag_equations_and_ports(
    ops: Sequence[OpSpec],
    *,
    preserve_port_ids: Sequence[str] = (),
) -> tuple[tuple[str, ...], tuple[FormulaTensorSpec, ...], tuple[FormulaTensorSpec, ...], tuple[str, ...]]:
    """Build composed symbolic equations, input/output tensor specs, and elided intermediates from true DAG."""
    ordered = tuple(ops)
    if not ordered:
        return (), (), (), ()
    member_ids = {op.op_id for op in ordered}
    preserve_set = set(preserve_port_ids)
    infos = tuple(_infer_op_symbolic_kind(op) for op in ordered)
    operand_ports: dict[str, tuple[tuple[str, str, int], ...]] = {}
    internal_users: dict[str, list[str]] = {op.op_id: [] for op in ordered}
    consumed_ports: dict[str, set[tuple[str, int]]] = {op.op_id: set() for op in ordered}
    unique_ext: list[tuple[tuple[str, str, int], OpSpec, bool]] = []
    seen_ext: set[tuple[str, str, int]] = set()

    for op, info in zip(ordered, infos):
        ports = tuple(_extract_op_operand_ports(op))
        if not ports:
            ports = ((f"__ext__:{op.op_id}", "output", 0),)
        operand_ports[op.op_id] = ports
        seen_int: set[str] = set()
        for prod_id, pname, ver in ports:
            if prod_id in member_ids:
                consumed_ports[prod_id].add((pname, ver))
                if prod_id not in seen_int:
                    internal_users[prod_id].append(op.op_id)
                    seen_int.add(prod_id)
            else:
                ek = (prod_id, pname, ver)
                if ek not in seen_ext:
                    seen_ext.add(ek)
                    unique_ext.append((ek, op, info[1]))

    ext_sym: dict[tuple[str, str, int], str] = {}
    in_specs: list[FormulaTensorSpec] = []
    for idx, (ek, op, is_mat) in enumerate(unique_ext):
        sym = "x" if len(unique_ext) == 1 else f"x{idx}"
        ext_sym[ek] = sym
        meta = dict(op.metadata) if op.metadata else {}
        entry = _typed_input_entry(op, ek[0], ek[1], ek[2])
        if entry is not None:
            in_dt = str(entry.get("dtype") or _resolve_op_input_dtype(op, meta))
            nbytes = float(entry.get("bytes", entry.get("nbytes", entry.get("byte_size", 0.0))) or 0.0)
            sh = _shape_from_dims_or_bytes(entry.get("shape"), nbytes, in_dt) or _op_input_shape(op, is_mat)
        else:
            sh = _op_input_shape(op, is_mat)
            in_dt = _resolve_op_input_dtype(op, meta)
        if sh:
            in_specs.append(FormulaTensorSpec(sym, sh, in_dt, "input", port=ek[1], version=ek[2]))

    live_out_ids = [
        op.op_id for op in ordered
        if (not internal_users[op.op_id]) or (op.op_id in preserve_set)
    ] or [ordered[-1].op_id]
    live_out_set = set(live_out_ids)

    sym_by_id: dict[str, str] = {}
    out_specs: list[FormulaTensorSpec] = []
    temp_idx = 1
    out_idx = 0
    for op, info in zip(ordered, infos):
        meta = dict(op.metadata) if op.metadata else {}
        if op.op_id in live_out_set:
            sym = "y" if len(live_out_ids) == 1 else f"y{out_idx}"
            out_idx += 1
            sym_by_id[op.op_id] = sym
            out_specs.extend(_op_declared_outputs(op, sym, info[1], meta))
        else:
            sym = f"t{temp_idx}"
            sym_by_id[op.op_id] = sym
            temp_idx += 1
            consumed = consumed_ports[op.op_id]
            for spec_out in _op_declared_outputs(op, sym, info[1], meta):
                if (spec_out.port, spec_out.version) not in consumed:
                    out_specs.append(spec_out)

    equations: list[str] = []
    for op, (kind, _is_mat, _is_red, act_kind, act_params) in zip(ordered, infos):
        in_syms_list: list[str] = []
        for prod_id, pname, ver in operand_ports[op.op_id]:
            if prod_id in member_ids:
                base_sym = sym_by_id[prod_id]
                if pname != "output" and ver:
                    in_syms_list.append(f"{base_sym}.{pname}@v{ver}")
                elif pname != "output":
                    in_syms_list.append(f"{base_sym}.{pname}")
                elif ver:
                    in_syms_list.append(f"{base_sym}@v{ver}")
                else:
                    in_syms_list.append(base_sym)
            else:
                in_syms_list.append(ext_sym[(prod_id, pname, ver)])
        meta = dict(op.metadata) if op.metadata else {}
        if "compute_dtype" not in meta and "weight_format" not in meta:
            p_abi = getattr(op, "precision_abi", None)
            resolved_dt = str(
                (getattr(p_abi, "compute_dtype", "") if p_abi is not None else "")
                or (getattr(p_abi, "weight_dtype", "") if p_abi is not None else "")
                or (op.matrix_flops[0][0] if op.matrix_flops else "")
                or op.dtype
                or ""
            ).strip()
            if resolved_dt:
                meta["op_dtype"] = resolved_dt
        equations.append(_format_dag_equation(
            op.op_id, op.local_shape, kind, tuple(in_syms_list), sym_by_id[op.op_id],
            output_dtype=_resolve_op_output_dtype(op, meta),
            activation_kind=act_kind,
            activation_params=act_params,
            group_size=int(meta.get("group_size", 0) or 0),
            collective_axis=str(meta.get("collective_axis") or ""),
            op_meta=meta,
        ))
    elided_ids = tuple(
        op.op_id for op in ordered
        if op.op_id not in live_out_set and internal_users[op.op_id]
    )
    return tuple(equations), tuple(in_specs), tuple(out_specs), elided_ids


def _resolve_reduction_order_from_ops(ops: Sequence[OpSpec]) -> str:
    """Resolve declared floating-point reduction order across absorbed atomic ops."""
    found: list[tuple[str, str]] = []
    for op in ops:
        meta = dict(op.metadata) if op.metadata else {}
        raw = str(meta.get("reduction_order") or "").strip()
        if raw:
            found.append((op.op_id, raw))
    if not found:
        return ""
    unique = tuple(dict.fromkeys(val for _, val in found))
    if len(unique) == 1:
        return unique[0]
    return ",".join(f"{oid}:{val}" for oid, val in found)


@dataclass(frozen=True, slots=True)
class FormulaKernelSpec:
    """Layer-1 declarative mathematical formula specification for unbound kernels."""

    name: str
    equations: tuple[str, ...]
    inputs: tuple[FormulaTensorSpec, ...] = ()
    outputs: tuple[FormulaTensorSpec, ...] = ()
    weights: tuple[FormulaTensorSpec, ...] = ()
    matrix_flops: tuple[tuple[str, float], ...] = ()
    vector_ops_fp32: float = 0.0
    link_bytes: float = 0.0
    row_reduction: bool = False
    uses_simt: bool = False
    atomic_ops: tuple[OpSpec, ...] = ()
    abi_signature_id: str = ""
    family: str = "formula_kernel"
    activation: str = ""
    activation_params: tuple[tuple[str, float], ...] = ()
    hbm_bytes_override: float | None = None
    mte2_bytes_override: float | None = None
    mte3_bytes_override: float | None = None
    pipe_stages: tuple[PipeStageSpec, ...] = ()
    reduction_order: str = ""

    def resolved_abi_signature(self) -> str:
        if self.abi_signature_id:
            return self.abi_signature_id
        payload = json.dumps(
            {
                "name": self.name,
                "equations": list(self.equations),
                "inputs": [(t.name, list(t.shape), t.dtype, t.port, int(t.version)) for t in self.inputs],
                "outputs": [(t.name, list(t.shape), t.dtype, t.port, int(t.version)) for t in self.outputs],
                "weights": [(t.name, list(t.shape), t.dtype, t.port, int(t.version)) for t in self.weights],
                "matrix_flops": [[dt, float(fl)] for dt, fl in self.matrix_flops],
                "row_reduction": bool(self.row_reduction),
                "uses_simt": bool(self.uses_simt),
                "reduction_order": str(self.reduction_order),
                "activation": self.activation,
                "activation_params": [[k, float(v)] for k, v in self.activation_params],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:ABI_DIGEST_LEN]
        return f"formula:{self.name}:{digest}"

    @classmethod
    def from_ops(
        cls,
        ops: Sequence[OpSpec],
        *,
        name: str = "",
        equations: Sequence[str] = (),
        saved_read_bytes: float = 0.0,
        saved_write_bytes: float = 0.0,
        preserve_port_ids: Sequence[str] = (),
        uses_simt: bool | None = None,
        reduction_order: str = "",
    ) -> FormulaKernelSpec:
        """Construct a ``FormulaKernelSpec`` directly from absorbed atomic ``OpSpec``s."""
        ordered = tuple(ops)
        if not ordered:
            raise ModelingError("formula_kernel_spec_empty_ops")
        infos = tuple(_infer_op_symbolic_kind(op) for op in ordered)
        kernel_name = name or ordered[-1].recipe or ordered[-1].op_id
        matrix: dict[str, float] = defaultdict(float)
        for op in ordered:
            entries = op.matrix_flops or (((op.dtype, op.flops),) if op.flops else ())
            for dt, fl in entries:
                if fl > 0.0:
                    matrix[str(dt)] += float(fl)
        raw_mte2 = sum(float(op.mte2_bytes or 0.0) for op in ordered)
        raw_mte3 = sum(float(op.mte3_bytes or 0.0) for op in ordered)
        raw_hbm = sum(float(op.hbm_bytes or 0.0) for op in ordered)
        mte2 = max(0.0, raw_mte2 - max(0.0, float(saved_read_bytes)))
        mte3 = max(0.0, raw_mte3 - max(0.0, float(saved_write_bytes)))
        hbm = max(0.0, raw_hbm - max(0.0, float(saved_read_bytes)) - max(0.0, float(saved_write_bytes)))
        vec_ops = sum(float(op.vector_ops or 0.0) for op in ordered)
        link_b = sum(float(op.link_bytes or 0.0) for op in ordered)
        row_red = any(info[2] for info in infos)
        acts = [info[3] for info in infos if info[3]]
        act_params = tuple(dict(p for info in infos for p in info[4]).items())
        dag_eqs, in_specs, out_specs, _elided = _compose_dag_equations_and_ports(
            ordered, preserve_port_ids=preserve_port_ids,
        )
        comp_eqs = tuple(equations) if equations else dag_eqs
        pipe_stages = _derive_pipe_stages_from_ops(ordered)
        if uses_simt is None:
            inferred_simt = any(
                bool(m.get("uses_simt"))
                or str(m.get("backend") or m.get("backend_target") or "").strip().lower()
                in {"tilelang_simt", "tla_simt"}
                for m in (dict(op.metadata) if op.metadata else {} for op in ordered)
            )
        else:
            inferred_simt = bool(uses_simt)
        eff_red_order = str(reduction_order or "").strip() or _resolve_reduction_order_from_ops(ordered)
        return cls(
            name=kernel_name,
            equations=comp_eqs,
            inputs=in_specs,
            outputs=out_specs,
            matrix_flops=tuple(sorted(matrix.items())),
            vector_ops_fp32=vec_ops,
            link_bytes=link_b,
            row_reduction=row_red,
            uses_simt=inferred_simt,
            atomic_ops=ordered,
            family="rewrite_fused",
            activation=acts[0] if acts else "",
            activation_params=act_params,
            hbm_bytes_override=hbm,
            mte2_bytes_override=mte2,
            mte3_bytes_override=mte3,
            pipe_stages=pipe_stages,
            reduction_order=eff_red_order,
        )


@dataclass(frozen=True, slots=True)
class FormulaKernelEvaluation:
    """White-box Engine + TileSolver evaluation receipt for a ``FormulaKernelSpec``."""

    name: str
    abi_signature_id: str
    composed_equations: tuple[str, ...]
    projected_us: float
    l_star_us: float
    tile_spec: TilePipelineSpec | None
    ub_peak_bytes: int
    ub_budget_bytes: int
    l1_peak_bytes: int
    l0_peak_bytes: int
    sram_verified: bool
    binding_state: BindingState = "unbound_formula_projection"
    has_unbound_formula_ops: bool = True
    dominant_pipe: str = ""
    hbm_bytes: float = 0.0
    mte2_bytes: float = 0.0
    mte3_bytes: float = 0.0
    launch_us: float = 0.0
    tile_algorithm: str = "resident_1pass"
    tile_shape_str: str = ""
    l1_budget_bytes: int = 0
    l0_budget_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "abi_signature_id": self.abi_signature_id,
            "composed_equations": list(self.composed_equations),
            "projected_us": round(float(self.projected_us), 4),
            "l_star_us": round(float(self.l_star_us), 4),
            "ub_peak_bytes": int(self.ub_peak_bytes),
            "ub_budget_bytes": int(self.ub_budget_bytes),
            "l1_peak_bytes": int(self.l1_peak_bytes),
            "l0_peak_bytes": int(self.l0_peak_bytes),
            "sram_verified": bool(self.sram_verified),
            "binding_state": self.binding_state,
            "has_unbound_formula_ops": bool(self.has_unbound_formula_ops),
            "dominant_pipe": self.dominant_pipe,
            "hbm_bytes": float(self.hbm_bytes),
            "mte2_bytes": float(self.mte2_bytes),
            "mte3_bytes": float(self.mte3_bytes),
            "launch_us": round(float(self.launch_us), 4),
        }


@dataclass(frozen=True, slots=True)
class SkuTargetFloors:
    """Calibrated L0 launch and interconnect floors extracted from active SKU."""

    launch_vec_us: float
    launch_cube_us: float
    launch_norm_us: float
    oneshot_floor_us: float
    mc2_sync_us: float
    is_intra_pod: bool
    is_a5_ub: bool
    interconnect_label: str


@dataclass(frozen=True, slots=True)
class KernelSynthesisSpec:
    """Composed symbolic math equations + Ascend FP32 UB/L1/L0 SRAM spec."""

    spec_id: str
    kernel_name: str
    abi_tier: str
    suggested_backend: str
    status_tag: str
    opt_pillar: str
    constituent_op_ids: tuple[str, ...]
    constituent_traits: tuple[str, ...]
    symbolic_equations: tuple[str, ...]
    input_tensors: tuple[str, ...]
    output_tensors: tuple[str, ...]
    elided_intermediates: tuple[str, ...]
    elided_hbm_bytes: int
    tile_algorithm: str
    tile_shape: str
    ub_precision_mode: str
    ub_peak_bytes: int
    ub_budget_bytes: int
    l1_peak_bytes: int
    l1_budget_bytes: int
    l0_peak_bytes: int
    l0_budget_bytes: int
    sram_verified: bool
    l_star_us: float
    before_us: float
    target_us: float
    target_attainment_pct: float
    abi_signature_id: str = ""
    binding_state: BindingState = "unbound_formula_projection"
    exposed_before_us: float | None = None
    ub_wall_warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "spec_id": self.spec_id,
            "kernel_name": self.kernel_name,
            "abi_tier": self.abi_tier,
            "suggested_backend": self.suggested_backend,
            "status_tag": self.status_tag,
            "opt_pillar": self.opt_pillar,
            "constituent_op_ids": list(self.constituent_op_ids),
            "constituent_traits": list(self.constituent_traits),
            "symbolic_equations": list(self.symbolic_equations),
            "input_tensors": list(self.input_tensors),
            "output_tensors": list(self.output_tensors),
            "elided_intermediates": list(self.elided_intermediates),
            "elided_hbm_bytes": int(self.elided_hbm_bytes),
            "tile_algorithm": self.tile_algorithm,
            "tile_shape": self.tile_shape,
            "ub_precision_mode": self.ub_precision_mode,
            "ub_peak_bytes": int(self.ub_peak_bytes),
            "ub_budget_bytes": int(self.ub_budget_bytes),
            "l1_peak_bytes": int(self.l1_peak_bytes),
            "l1_budget_bytes": int(self.l1_budget_bytes),
            "l0_peak_bytes": int(self.l0_peak_bytes),
            "l0_budget_bytes": int(self.l0_budget_bytes),
            "sram_verified": bool(self.sram_verified),
            "l_star_us": round(float(self.l_star_us), 3),
            "before_us": round(float(self.before_us), 3),
            "exposed_before_us": round(
                float(self.exposed_before_us if self.exposed_before_us is not None else self.before_us), 3,
            ),
            "target_us": round(float(self.target_us), 3),
            "target_attainment_pct": round(float(self.target_attainment_pct), 1),
            "abi_signature_id": self.abi_signature_id,
            "binding_state": self.binding_state,
        }
        if self.ub_wall_warning:
            payload["ub_wall_warning"] = self.ub_wall_warning
        return payload

    def format_formula_block(self) -> str:
        eq_txt = " ; ".join(self.symbolic_equations)
        ub_kb = self.ub_peak_bytes / KIB_BYTES
        ub_cap_kb = self.ub_budget_bytes / KIB_BYTES
        l1_kb = self.l1_peak_bytes / KIB_BYTES
        l1_cap_kb = self.l1_budget_bytes / KIB_BYTES
        elided_kb = self.elided_hbm_bytes / KIB_BYTES
        warn_suffix = f" | WARN: {self.ub_wall_warning}" if self.ub_wall_warning else ""
        return (
            f"Math: {eq_txt} | "
            f"Tile: {self.tile_algorithm} {self.tile_shape} ({self.ub_precision_mode}) | "
            f"UB={ub_kb:.1f}/{ub_cap_kb:.1f}KiB, L1={l1_kb:.1f}/{l1_cap_kb:.1f}KiB "
            f"({'PASS' if self.sram_verified else 'FAIL'}) | "
            f"Elided HBM={elided_kb:.1f}KiB{warn_suffix}"
        )
