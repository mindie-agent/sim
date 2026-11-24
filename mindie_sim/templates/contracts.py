"""Immutable template specializations and whole-step accounting contracts.

[Role] Define the public catalog, compiler and certificate interface records.
[Invariants] Provenance pins blobs and symbols. Byte ranges are half-open.
[Timeouts & Locks] Validation and composition use local data without locks.
[Side-Effect Policy] No device, repository or filesystem mutation.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
from graphlib import CycleError, TopologicalSorter
from math import isfinite, lcm
from typing import Iterable, Literal

from mindie_sim.prediction.program.stages import KernelProgram
from mindie_sim.prediction.runtime.memory_physics import (
    AddressRun, SectorCoverage, _MAX_EXPLICIT_SECTOR_INTERVALS, _merge_byte_intervals,
)
from mindie_sim.prediction.runtime.l2_residency import (
    _coalesce_runs, _coverage_runs, _partition_runs, _range_run, _residency_runs, _run_end,
)
from mindie_sim.prediction.runtime.tile_solver import AlgorithmKind, TileSolution
from mindie_sim.schema.abi import SOC_ARCHS, SocArch
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef

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


def attn_res_tiling(tokens, aiv, ub_bytes):
    del tokens, aiv, ub_bytes
    raise NotImplementedError(
        "LEGACY_PURGED: pass explicit hidden_size into attn_res_tiling without hardcoded 7168 fingerprint"
    )


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
    # A live program publishes the latest completed invocation's delivery.
    realized_hbm_bytes: float | None = field(default=None, compare=False)
    realized_l2_miss_bytes: int = field(default=0, compare=False)

    @property
    def l2_hit_bytes(self) -> int:
        return self.realized_l2_hit_bytes

    @property
    def l2_miss_bytes(self) -> int:
        return self.realized_l2_miss_bytes

    def _record_delivery(self, delivered, reads, writes):
        object.__setattr__(self, "read_obligations", reads)
        object.__setattr__(self, "write_obligations", writes)
        for name in ("logical_transfer_bytes", "issued_copy_bytes", "dram_sector_bytes", "turnaround_service_bytes",
                     "realized_l2_hit_bytes", "realized_l2_miss_bytes", "realized_hbm_bytes"):
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
