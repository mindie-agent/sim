"""Typed physical mechanism declarations shared by planning and execution.

[Role] Carry the memory hierarchy, matrix policy, attention/indexer/vector and
  collective declarations that turn tensor work into finite physical stages.
[Invariants] A mechanism record is immutable and serializable. It never carries
  measured latency, model labels or a fallback rate. Device identity includes
  the complete signature, so program and mapping caches cannot alias two
  physical declarations.
[Timeouts & Locks] Pure parsing of bounded mappings; no I/O, locks or devices.
[Side-Effect Policy] Returns new immutable records; caller mappings are copied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .common import ModelingError


ZERO = 0
ONE = 1
FIRST = 0
SECOND = 1
PAIR_LEN = 2
TRIPLE_LEN = 3
DEFAULT_CACHE_LINE_BYTES = 128
DEFAULT_L2_EFFECTIVE_FRACTION = 0.75
DEFAULT_BASE_TILES = ((128, 128), (128, 256), (256, 128), (256, 256))
DEFAULT_REFILL_K_TILES = (64, 128, 256)
DEFAULT_SPLIT_K_CANDIDATES = (1, 2, 4, 8, 16, 32)
DEFAULT_MIN_K_PER_SPLIT = 64
DEFAULT_INPUT_SLOTS = 2
DEFAULT_COMPUTE_SLOTS = 2
DEFAULT_SWIZZLE_WINDOWS = ((1, 1), (2, 8), (4, 8))
DEFAULT_OUTPUT_PATHS = ("gm", "l0c_to_ub", "onchip_reduce")
DEFAULT_MICRO_TILES = {
    "bf16": ((128, 128, 128), (64, 64, 128), (16, 16, 16)),
    "fp16": ((128, 128, 128), (64, 64, 128), (16, 16, 16)),
    "fp32": ((128, 128, 64), (64, 64, 64), (16, 16, 16)),
    "int8": ((128, 128, 128), (64, 64, 128), (16, 16, 32)),
    "fp8": ((128, 128, 128), (64, 64, 128), (16, 16, 32)),
}


def _frozen_mapping(value) -> Mapping:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ModelingError("physical_mechanism_mapping_required")
    return dict(value)


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or int(value) < ONE:
        raise ModelingError(f"physical_mechanism_positive_integer_required:{name}")
    return int(value)


def _fraction(value, name: str) -> float:
    parsed = float(value)
    if parsed <= ZERO or parsed > ONE:
        raise ModelingError(f"physical_mechanism_fraction_invalid:{name}")
    return parsed


def _pairs(value, name: str) -> tuple[tuple[int, int], ...]:
    result = []
    for item in value:
        if len(item) != PAIR_LEN:
            raise ModelingError(f"physical_mechanism_pair_required:{name}")
        result.append((_positive_int(item[FIRST], name), _positive_int(item[SECOND], name)))
    if not result:
        raise ModelingError(f"physical_mechanism_pair_list_empty:{name}")
    return tuple(result)


def _triples(value, name: str) -> tuple[tuple[int, int, int], ...]:
    result = []
    for item in value:
        if len(item) != TRIPLE_LEN:
            raise ModelingError(f"physical_mechanism_triple_required:{name}")
        result.append(tuple(_positive_int(part, name) for part in item))
    if not result:
        raise ModelingError(f"physical_mechanism_triple_list_empty:{name}")
    return tuple(result)


@dataclass(frozen=True)
class MemoryHierarchySpec:
    l2_capacity_bytes: int
    l2_effective_fraction: float
    cache_line_bytes: int
    write_allocate: bool
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "l2_capacity_bytes",
                           max(ZERO, int(self.l2_capacity_bytes)))
        object.__setattr__(self, "cache_line_bytes",
                           _positive_int(self.cache_line_bytes, "cache_line_bytes"))
        object.__setattr__(self, "l2_effective_fraction",
                           _fraction(self.l2_effective_fraction, "l2_effective_fraction"))
        object.__setattr__(self, "provenance", _frozen_mapping(self.provenance))

    @property
    def effective_l2_bytes(self) -> float:
        return float(self.l2_capacity_bytes) * self.l2_effective_fraction

    def to_dict(self) -> dict:
        return {
            "l2_capacity_bytes": self.l2_capacity_bytes,
            "l2_effective_fraction": self.l2_effective_fraction,
            "cache_line_bytes": self.cache_line_bytes,
            "write_allocate": self.write_allocate,
            "provenance": dict(self.provenance),
        }

    @property
    def signature(self):
        return (self.l2_capacity_bytes, self.l2_effective_fraction,
                self.cache_line_bytes, self.write_allocate,
                tuple(sorted((str(k), repr(v)) for k, v in self.provenance.items())))


@dataclass(frozen=True)
class MatrixPolicySpec:
    base_tiles: tuple[tuple[int, int], ...]
    refill_k_tiles: tuple[int, ...]
    micro_tiles_by_dtype: Mapping[str, tuple[tuple[int, int, int], ...]]
    input_slots: int
    compute_slots: int
    split_k_candidates: tuple[int, ...]
    min_k_per_split: int
    swizzle_windows: tuple[tuple[int, int], ...]
    supported_output_paths: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "base_tiles", _pairs(self.base_tiles, "base_tiles"))
        refills = tuple(_positive_int(value, "refill_k_tiles")
                        for value in self.refill_k_tiles)
        object.__setattr__(self, "refill_k_tiles", refills)
        micro = {str(key): _triples(value, f"micro_tiles:{key}")
                 for key, value in dict(self.micro_tiles_by_dtype).items()}
        object.__setattr__(self, "micro_tiles_by_dtype", micro)
        object.__setattr__(self, "input_slots",
                           _positive_int(self.input_slots, "input_slots"))
        object.__setattr__(self, "compute_slots",
                           _positive_int(self.compute_slots, "compute_slots"))
        splits = tuple(sorted({_positive_int(value, "split_k_candidates")
                               for value in self.split_k_candidates}))
        object.__setattr__(self, "split_k_candidates", splits)
        object.__setattr__(self, "min_k_per_split",
                           _positive_int(self.min_k_per_split, "min_k_per_split"))
        object.__setattr__(self, "swizzle_windows",
                           _pairs(self.swizzle_windows, "swizzle_windows"))
        paths = tuple(str(value) for value in self.supported_output_paths)
        if not paths:
            raise ModelingError("physical_mechanism_output_paths_empty")
        object.__setattr__(self, "supported_output_paths", paths)

    @property
    def signature(self):
        return (
            tuple(self.base_tiles), tuple(self.refill_k_tiles),
            tuple((key, tuple(value)) for key, value in
                  sorted(self.micro_tiles_by_dtype.items())),
            self.input_slots, self.compute_slots,
            tuple(self.split_k_candidates), self.min_k_per_split,
            tuple(self.swizzle_windows), tuple(self.supported_output_paths),
        )

    def to_dict(self) -> dict:
        return {
            "base_tiles": [list(value) for value in self.base_tiles],
            "refill_k_tiles": list(self.refill_k_tiles),
            "micro_tiles_by_dtype": {key: [list(value) for value in values]
                                     for key, values in self.micro_tiles_by_dtype.items()},
            "input_slots": self.input_slots,
            "compute_slots": self.compute_slots,
            "split_k_candidates": list(self.split_k_candidates),
            "min_k_per_split": self.min_k_per_split,
            "swizzle_windows": [list(value) for value in self.swizzle_windows],
            "supported_output_paths": list(self.supported_output_paths),
        }


@dataclass(frozen=True)
class PhysicalMechanismSpec:
    memory_hierarchy: MemoryHierarchySpec
    matrix: MatrixPolicySpec
    attention: Mapping[str, object] = field(default_factory=dict)
    indexer: Mapping[str, object] = field(default_factory=dict)
    vector: Mapping[str, object] = field(default_factory=dict)
    collective: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("attention", "indexer", "vector", "collective"):
            object.__setattr__(self, name, _frozen_mapping(getattr(self, name)))

    @property
    def signature(self):
        return (
            self.memory_hierarchy.signature, self.matrix.signature,
            tuple((name, tuple(sorted((str(k), repr(v)) for k, v in
                                      getattr(self, name).items())))
                  for name in ("attention", "indexer", "vector", "collective")),
        )

    def to_dict(self) -> dict:
        return {
            "memory_hierarchy": self.memory_hierarchy.to_dict(),
            "matrix": self.matrix.to_dict(),
            "attention": dict(self.attention),
            "indexer": dict(self.indexer),
            "vector": dict(self.vector),
            "collective": dict(self.collective),
        }


def default_physical_mechanisms(*, l2_capacity_bytes: int = ZERO) -> PhysicalMechanismSpec:
    """Conservative declaration for a legacy profile without typed mechanisms."""
    return PhysicalMechanismSpec(
        MemoryHierarchySpec(
            l2_capacity_bytes=max(ZERO, int(l2_capacity_bytes)),
            l2_effective_fraction=DEFAULT_L2_EFFECTIVE_FRACTION,
            cache_line_bytes=DEFAULT_CACHE_LINE_BYTES,
            write_allocate=True,
            provenance={"basis": "legacy_profile_default", "calibrated": False},
        ),
        MatrixPolicySpec(
            base_tiles=DEFAULT_BASE_TILES,
            refill_k_tiles=DEFAULT_REFILL_K_TILES,
            micro_tiles_by_dtype=DEFAULT_MICRO_TILES,
            input_slots=DEFAULT_INPUT_SLOTS,
            compute_slots=DEFAULT_COMPUTE_SLOTS,
            split_k_candidates=DEFAULT_SPLIT_K_CANDIDATES,
            min_k_per_split=DEFAULT_MIN_K_PER_SPLIT,
            swizzle_windows=DEFAULT_SWIZZLE_WINDOWS,
            supported_output_paths=DEFAULT_OUTPUT_PATHS,
        ),
    )


def _pairs_from(value, default):
    if value is None:
        return default
    return tuple(tuple(item) for item in value)


def parse_physical_mechanisms(value, *, fallback_l2_bytes: int = ZERO) -> PhysicalMechanismSpec:
    """Parse one device declaration, retaining strict typed validation."""
    if value is None:
        return default_physical_mechanisms(l2_capacity_bytes=fallback_l2_bytes)
    if not isinstance(value, Mapping):
        raise ModelingError("device_physical_mechanisms_requires_mapping")
    memory = value.get("memory_hierarchy") or {}
    matrix = value.get("matrix") or {}
    if not isinstance(memory, Mapping) or not isinstance(matrix, Mapping):
        raise ModelingError("device_physical_mechanisms_sections_require_mapping")
    hierarchy = MemoryHierarchySpec(
        l2_capacity_bytes=int(memory.get("l2_capacity_bytes", fallback_l2_bytes)),
        l2_effective_fraction=float(memory.get(
            "l2_effective_fraction", DEFAULT_L2_EFFECTIVE_FRACTION)),
        cache_line_bytes=int(memory.get("cache_line_bytes", DEFAULT_CACHE_LINE_BYTES)),
        write_allocate=bool(memory.get("write_allocate", True)),
        provenance=memory.get("provenance") or {},
    )
    policy = MatrixPolicySpec(
        base_tiles=_pairs_from(matrix.get("base_tiles"), DEFAULT_BASE_TILES),
        refill_k_tiles=tuple(matrix.get("refill_k_tiles") or DEFAULT_REFILL_K_TILES),
        micro_tiles_by_dtype=matrix.get("micro_tiles_by_dtype") or DEFAULT_MICRO_TILES,
        input_slots=int(matrix.get("input_slots", DEFAULT_INPUT_SLOTS)),
        compute_slots=int(matrix.get("compute_slots", DEFAULT_COMPUTE_SLOTS)),
        split_k_candidates=tuple(matrix.get("split_k_candidates")
                                 or DEFAULT_SPLIT_K_CANDIDATES),
        min_k_per_split=int(matrix.get("min_k_per_split", DEFAULT_MIN_K_PER_SPLIT)),
        swizzle_windows=_pairs_from(matrix.get("swizzle_windows"),
                                    DEFAULT_SWIZZLE_WINDOWS),
        supported_output_paths=tuple(matrix.get("supported_output_paths")
                                     or DEFAULT_OUTPUT_PATHS),
    )
    return PhysicalMechanismSpec(
        hierarchy, policy,
        attention=value.get("attention") or {},
        indexer=value.get("indexer") or {},
        vector=value.get("vector") or {},
        collective=value.get("collective") or {},
    )
