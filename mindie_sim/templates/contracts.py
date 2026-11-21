"""Immutable template specializations and whole-step accounting contracts.

[Role] Define the public catalog, compiler and certificate interface records.
[Invariants] Provenance pins blobs and symbols. Byte ranges are half-open.
[Timeouts & Locks] Validation and composition use local data without locks.
[Side-Effect Policy] No device, repository or filesystem mutation.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from graphlib import CycleError, TopologicalSorter
from math import isfinite
from typing import Iterable, Literal

from mindie_sim.prediction.program.stages import KernelProgram
from mindie_sim.prediction.runtime.tile_solver import AlgorithmKind, TileSolution
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef

COMMIT_HEX_LENGTH = 40
SHA256_HEX_LENGTH = 64
LEGACY_MATRIX_ARCH_IDS = frozenset({"npu_arch_2201", "arch22"})
DIRECT_MATRIX_ARCH_IDS = frozenset({"npu_arch_3510", "arch35"})

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

ATTN_RES_HIDDEN_SIZE = 7168
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


def attn_res_tiling(tokens, aiv, ub_bytes=ATTN_RES_UB_BYTES):
    if min(tokens, aiv) < 1:
        raise ModelingError("attn_res_problem_invalid")
    rows = ceil_div(tokens, aiv)
    d = ATTN_RES_HIDDEN_SIZE
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
    source_refs: tuple[TemplateSourceRef, ...]

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
    source_refs: tuple[TemplateSourceRef, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TensorRangeObligation:
    die: int
    allocation_id: str
    tensor_version: int
    ranges: tuple[tuple[int, int], ...]
    initial_residency: Literal["hbm_cold", "l2_warm", "onchip_resident"]
    required_final_visibility: Literal["owner", "die", "device", "host"]

    def __post_init__(self) -> None:
        if (
            type(self.die) is not int or self.die < 0 or not self.allocation_id
            or type(self.tensor_version) is not int or self.tensor_version < 0
            or self.initial_residency not in {"hbm_cold", "l2_warm", "onchip_resident"}
            or self.required_final_visibility not in {"owner", "die", "device", "host"}
        ):
            raise ModelingError("tensor_range_obligation_invalid")
        end = 0
        for lo, hi in self.ranges:
            if type(lo) is not int or type(hi) is not int or not end <= lo < hi:
                raise ModelingError(f"tensor_range_intervals_invalid:{self.allocation_id}")
            end = hi


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
class FuseRegionSpec:
    region_id: str
    op_ids: tuple[str, ...]
    live_in_obligations: tuple[TensorRangeObligation, ...]
    live_out_obligations: tuple[TensorRangeObligation, ...]
    edge_routes: tuple[EdgeRouteBinding, ...]
    reduction_class: AlgorithmKind
    tile_solution: TileSolution
    backend: Literal["catlass_evg", "dsl_mix", "dsl_vec", "tla_simt"]


@dataclass(frozen=True, slots=True)
class SemanticWorkItem:
    work_id: str
    kind: Literal["matrix_mac", "vector_alu", "vector_sfu", "vector_reduce", "sort_topk"]
    dtype: str
    logical_ops: int
    admitted_engines: tuple[Literal["cube", "aiv_simd", "aiv_simt"], ...]
    admissible_schedulers: tuple[str, ...]

    @property
    def eligible_engines(self) -> tuple[str, ...]:
        return self.admitted_engines

    def __post_init__(self) -> None:
        if (
            not self.work_id or not self.dtype or type(self.logical_ops) is not int or self.logical_ops < 0
            or not self.admitted_engines or not set(self.admitted_engines) <= {"cube", "aiv_simd", "aiv_simt"}
            or not self.admissible_schedulers
            or self.kind not in {"matrix_mac", "vector_alu", "vector_sfu", "vector_reduce", "sort_topk"}
        ):
            raise ModelingError(f"template_semantic_work_invalid:{self.work_id}")


def eligible_matrix_engines(problem, datapath) -> tuple[str, ...]:
    if problem.indices is not None or problem.physical_m is not None:
        return ("aiv_simt",)
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


@dataclass(frozen=True, slots=True)
class TemplateLoweringResult:
    program: KernelProgram
    summary: FusedRegionSummary
    stages_count: int
    l2_hit_bytes: int
    dram_miss_bytes: int


def _union_obligations(
    obligations: Iterable[TensorRangeObligation],
) -> tuple[TensorRangeObligation, ...]:
    grouped: dict[tuple[int, str, int], list[TensorRangeObligation]] = {}
    for obligation in obligations:
        key = (obligation.die, obligation.allocation_id, obligation.tensor_version)
        grouped.setdefault(key, []).append(obligation)
    visibility = ("owner", "die", "device", "host")
    residency = ("hbm_cold", "l2_warm", "onchip_resident")
    result = []
    for key, entries in sorted(grouped.items()):
        events = {}
        for entry in entries:
            attributes = (residency.index(entry.initial_residency), visibility.index(entry.required_final_visibility))
            for lo, hi in entry.ranges:
                events.setdefault(lo, []).append((attributes, 1))
                events.setdefault(hi, []).append((attributes, -1))
        active, merged = {}, {}
        previous = 0
        for offset, changes in sorted(events.items()):
            if active:
                state = min(r for r, _ in active), max(v for _, v in active)
                ranges = merged.setdefault(state, [])
                if ranges and ranges[-1][1] == previous:
                    ranges[-1] = (ranges[-1][0], offset)
                else:
                    ranges.append((previous, offset))
            for attributes, delta in changes:
                count = active.get(attributes, 0) + delta
                if count:
                    active[attributes] = count
                else:
                    active.pop(attributes, None)
            previous = offset
        result.extend(TensorRangeObligation(*key, tuple(ranges), residency[r], visibility[v])
                      for (r, v), ranges in merged.items())
    return tuple(result)


def compose_step_obligations(
    summaries: Iterable[FusedRegionSummary],
) -> tuple[tuple[TensorRangeObligation, ...], tuple[TensorRangeObligation, ...]]:
    """Return canonical compulsory step reads and live-out writes, respectively."""
    summaries = tuple(summaries)
    return (
        _union_obligations(
            entry for summary in summaries for entry in summary.read_obligations
            if entry.initial_residency == "hbm_cold"
        ),
        _union_obligations(
            entry for summary in summaries for entry in summary.write_obligations
            if entry.required_final_visibility in {"device", "host"}
        ),
    )


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
