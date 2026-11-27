"""Contracts shared by constructive search and lower-bound assembly.

[Role] Carry semantic work, memory obligations and physically bound kernel specs.
[Invariants] Work and tensor ranges belong to explicit physical scopes.
[Timeouts & Locks] Value construction and finite lookups take no locks.
[Side-Effect Policy] Immutable records with no I/O or runtime initialization.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Protocol

from mindie_sim.primitives.schema import PrimitiveTable
from mindie_sim.schema.datapath import DatapathCapabilitySpec
from mindie_sim.templates.contracts import (
    CatlassKernelSpec, DslKernelSpec, EdgeRouteBinding, EvgVisitorNode, FusedRegionSummary,
    FuseRegionSpec, SemanticWorkItem, TensorRangeObligation,
)

__all__ = (
    "TensorRangeObligation", "SemanticWorkItem", "EvgVisitorNode",
    "CatlassKernelSpec", "DslKernelSpec", "EdgeRouteBinding", "FuseRegionSpec",
    "FusedRegionSummary", "BilinearContractionSpec", "PrimitiveCapacityEntry",
    "PrimitiveCapacityTable", "UpperCapacityProvider", "normalize_capacity_per_second",
)


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
    # The full (i, k) and (j, k) index sets map injectively within A and B.
    address_injective_across_k: bool = False


_GIGA = 1_000_000_000
_GIBI = 1024 * 1024 * 1024
_CAPACITY_UNIT_SCALE = {
    **dict.fromkeys(("b/s", "byte/s", "bytes/s", "flop/s", "flops/s", "ops/s", "elements/s"), 1),
    **dict.fromkeys(("b/ms", "byte/ms", "bytes/ms", "flop/ms", "flops/ms", "ops/ms",
                     "elements/ms", "kb/s"), 1000),
    **dict.fromkeys(("gb/s", "gflop/s", "gops/s"), _GIGA),
    **dict.fromkeys(("tb/s", "tflop/s", "tops/s"), 1000 * _GIGA),
    "mb/s": 1000 * 1000,
    "kib/s": 1024,
    "mib/s": 1024 * 1024,
    "gib/s": _GIBI,
    "tib/s": 1024 * _GIBI,
}


def normalize_capacity_per_second(value: float, unit: str) -> Fraction:
    """Convert a byte or operation rate to its base unit per second."""
    scale = _CAPACITY_UNIT_SCALE.get(unit.strip().lower())
    if scale is None:
        raise ValueError(f"unknown primitive capacity unit: {unit!r}")
    return Fraction(str(value)) * scale


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
    """Raw capacities need units from `unit()` or entries, unless a per-second lookup is supplied."""

    @property
    def sku(self) -> str: ...

    def upper_capacity(self, primitive_id: str) -> float | None: ...


def _provider_capacity_per_second(table: UpperCapacityProvider, primitive_id: str) -> Fraction | None:
    normalized = getattr(table, "upper_capacity_per_second", None)
    if callable(normalized):
        return normalized(primitive_id)
    capacity = table.upper_capacity(primitive_id)
    if capacity is None:
        return None
    unit_lookup = getattr(table, "unit", None)
    unit = unit_lookup(primitive_id) if callable(unit_lookup) else None
    if unit is None:
        entries = getattr(table, "entries", ())
        if isinstance(entries, Mapping):
            entries = entries.values()
        unit = next((entry.unit for entry in entries if entry.primitive_id == primitive_id), None)
    return normalize_capacity_per_second(capacity, unit) if unit is not None else None


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

    def upper_capacity_per_second(self, primitive_id: str) -> Fraction | None:
        return next((normalize_capacity_per_second(entry.certified_upper_capacity, entry.unit)
                     for entry in self.entries if entry.primitive_id == primitive_id), None)


CapacityTable = PrimitiveTable | UpperCapacityProvider

# Table rates are SI. Per-owner metrics must be aggregated over the physical die.
_HARDWARE_RATE_METRICS = {
    "hbm_bus_r": (("P01_datacopy", "hbm_read_bytes_per_s", None),),
    "hbm_bus_w": (("P01_datacopy", "hbm_write_bytes_per_s", None),),
    "mte2_total": (
        ("P01_datacopy", "mte2_aic_hit_bytes_per_s_per_core", "aic_count"),
        ("P01_datacopy", "mte2_aiv_hit_bytes_per_s_per_core", "aiv_count"),
    ),
    "write_total": (
        ("P03_fixpipe", "fixpipe_out_bytes_per_s_per_aic", "aic_count"),
        ("P04_mte3_store", "mte3_store_bytes_per_s_per_aiv", "aiv_count"),
    ),
    "l2_bw": (("P05_l2_residency_reuse", "l2_die_bytes_per_s", None),),
    "cube": (("P06_mmad", "cube_only_rate", "aic_count"),),
    "vector": (("P07_vec_alu", "vec_alu_ops_per_s_per_aiv", "aiv_count"),),
    "simt": (("P11_simt", "simt_ops_per_s_per_aiv", "aiv_count"),),
}
_MMAD_MODES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
               "w4a8": "int8", "a8w4_mx": "mxfp8", "fp8": "fp8_e4m3"}


def certified_capacity_per_second(
    table: CapacityTable, pipe: str, dtype: str, path: DatapathCapabilitySpec | None,
) -> Fraction | None:
    if not isinstance(table, PrimitiveTable):
        candidates = ([f"{pipe}:{dtype}"] if dtype else []) + [pipe]
        if pipe == "hbm_bus_r":
            candidates.append("P01_hbm_read")
        for primitive in candidates:
            capacity = _provider_capacity_per_second(table, primitive)
            if capacity is not None:
                return capacity
        return None
    if path is None or table.arch_preset != path.arch_id:
        return None
    metrics = _HARDWARE_RATE_METRICS.get(pipe)
    if metrics is None:
        return None
    total = Fraction()
    for primitive, metric, owners in metrics:
        count = getattr(path, owners) if owners else 1
        if not count:
            continue
        cell = table.cell(primitive)
        mode = (dtype if dtype in cell.modes else _MMAD_MODES.get(dtype, dtype)) if pipe == "cube" else None
        if (mode is not None and mode not in cell.modes) or cell.mode_status(mode) == "unsupported":
            return None
        total += Fraction(str(table.certified_upper_capacity(primitive, metric, mode=mode))) * count
    return total if total > 0 else None
