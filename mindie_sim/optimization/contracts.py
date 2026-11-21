"""Contracts shared by constructive search and lower-bound assembly.

[Role] Carry semantic work, memory obligations and physically bound kernel specs.
[Invariants] Work and tensor ranges belong to explicit physical scopes.
[Timeouts & Locks] Value construction and finite lookups take no locks.
[Side-Effect Policy] Immutable records with no I/O or runtime initialization.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Protocol

from mindie_sim.prediction.runtime.memory_physics import AddressRun
from mindie_sim.prediction.runtime.tile_solver import AlgorithmKind, TileSolution
from mindie_sim.schema.datapath import SourceRef

__all__ = (
    "TensorRangeObligation", "SemanticWorkItem", "EvgVisitorNode",
    "CatlassKernelSpec", "DslKernelSpec", "EdgeRouteBinding", "FuseRegionSpec",
    "FusedRegionSummary", "BilinearContractionSpec", "PrimitiveCapacityEntry",
    "PrimitiveCapacityTable", "UpperCapacityProvider",
)


@dataclass(frozen=True, slots=True)
class TensorRangeObligation:
    die: int
    allocation_id: str
    tensor_version: int
    ranges: tuple[tuple[int, int], ...]
    initial_residency: Literal["hbm_cold", "l2_warm", "onchip_resident"]
    required_final_visibility: Literal["owner", "die", "device", "host"]
    address_runs: tuple[AddressRun, ...] = ()


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


@dataclass(frozen=True, slots=True)
class EvgVisitorNode:
    node_id: str
    kind: Literal[
        "acc_load", "aux_load", "row_broadcast", "col_broadcast",
        "exp", "relu", "sqrt", "rsqrt_fast", "leaky_relu", "silu",
        "cast", "mul", "muls", "add", "aux_store",
    ]
    dtype: str
    layout: str = "RowMajor"
    inputs: tuple[str, ...] = ()
    aux_tensor_id: str | None = None
    scalar_param: float | None = None


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
    source_refs: tuple[SourceRef, ...]


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
    source_refs: tuple[SourceRef, ...]


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
class BilinearContractionSpec:
    contraction_id: str
    die: int
    m: int
    n: int
    k_red: int
    bits_a: int
    bits_b: int
    bits_acc: int
    pool_capacity_bytes: int
    crosses_cut: bool = True
    pair_injective: bool = True
    direct_mac_no_recomp: bool = True
    cold_entry: bool = True
    write_free: bool = False
    kind: str = "gemm"
    replica: int = 0
    pp_stage: int = 0
    distinct_operands: bool = False


@dataclass(frozen=True, slots=True)
class PrimitiveCapacityEntry:
    primitive_id: str
    sim_rate_p50: float
    certified_upper_capacity: float
    unit: str

    def __post_init__(self) -> None:
        if not 0 < self.sim_rate_p50 <= self.certified_upper_capacity < float("inf"):
            raise ValueError(f"invalid primitive capacity for {self.primitive_id}")


class UpperCapacityProvider(Protocol):
    @property
    def sku(self) -> str: ...

    def upper_capacity(self, primitive_id: str) -> float | None: ...


@dataclass(frozen=True, slots=True)
class PrimitiveCapacityTable:
    sku: str
    entries: tuple[PrimitiveCapacityEntry, ...]

    def __post_init__(self) -> None:
        if len({entry.primitive_id for entry in self.entries}) != len(self.entries):
            raise ValueError("duplicate primitive capacity entry")

    def upper_capacity(self, primitive_id: str) -> float | None:
        return next((entry.certified_upper_capacity for entry in self.entries
                     if entry.primitive_id == primitive_id), None)
