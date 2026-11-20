"""Immutable template specializations and whole-step accounting contracts.

[Role] Define the public catalog, compiler and certificate interface records.
[Invariants] Provenance pins blobs and symbols. Byte ranges are half-open.
[Timeouts & Locks] Validation and composition use local data without locks.
[Side-Effect Policy] No device, repository or filesystem mutation.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
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


def ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


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

    def __post_init__(self) -> None:
        if (
            not self.work_id or not self.dtype or type(self.logical_ops) is not int or self.logical_ops < 0
            or not self.admitted_engines or not set(self.admitted_engines) <= {"cube", "aiv_simd", "aiv_simt"}
            or not self.admissible_schedulers
            or self.kind not in {"matrix_mac", "vector_alu", "vector_sfu", "vector_reduce", "sort_topk"}
        ):
            raise ModelingError(f"template_semantic_work_invalid:{self.work_id}")


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
        merged: list[tuple[int, int]] = []
        for lo, hi in sorted(interval for entry in entries for interval in entry.ranges):
            if merged and lo <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))
        if merged:
            result.append(TensorRangeObligation(
                *key, tuple(merged),
                min((entry.initial_residency for entry in entries), key=residency.index),
                max((entry.required_final_visibility for entry in entries), key=visibility.index),
            ))
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
