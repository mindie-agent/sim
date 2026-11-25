"""Typed physical mechanism declarations shared by planning and execution.

[Role] Carry the memory hierarchy, matrix policy, attention/indexer/vector and
  collective declarations that turn tensor work into finite physical stages.
[Invariants] A mechanism record is immutable and serializable. It never carries
  measured latency, model labels or a fallback rate. Device identity includes
  the complete signature, so program and mapping caches cannot alias two
  physical declarations. Output policies are capability-validated against the
  declared ``ComputeArchSpec`` (``l0c_to_ub`` requires a declared ``l0c->ub``
  route), and shared L2 enforces ``buffer_units.l2 == 1``.
[Timeouts & Locks] Pure parsing of bounded mappings; no I/O, locks or devices.
[Side-Effect Policy] Returns new immutable records; caller mappings are copied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .common import ModelingError


PAIR_LEN = 2
TRIPLE_LEN = 3
DEFAULT_CACHE_LINE_BYTES = 128
DEFAULT_L2_EFFECTIVE_FRACTION = 0.75
DAV_2201_PRESET = "npu_arch_2201"
DAV_3510_PRESET = "npu_arch_3510"
DAV_3510_BODY_CANDIDATES = (
    (256, 256, 64),
    (128, 256, 128),
    (256, 128, 128),
    (128, 256, 64),
    (128, 128, 128),
    (128, 128, 64),
    (64, 256, 128),
    (64, 128, 128),
    (32, 128, 256),
    (32, 128, 128),
    (32, 64, 256),
    (32, 32, 256),
    (16, 128, 256),
    (16, 64, 256),
)
DEFAULT_BASE_TILES = ((128, 128), (128, 256), (256, 128), (256, 256))
DAV_2201_BASE_TILES = ((128, 128), (128, 256), (256, 128))
DEFAULT_REFILL_K_TILES = (64, 128, 256)
DAV_2201_REFILL_K_TILES = (64, 128, 256)
DEFAULT_MAPPED_REFILL_K_TILES = (64, 128)
DAV_2201_MAPPED_REFILL_K_TILES = (64, 128)
DEFAULT_SPLIT_K_CANDIDATES = (1, 2, 4, 8, 16, 32)
DAV_2201_SPLIT_K_CANDIDATES = (1, 2, 4, 8, 16, 24)
DEFAULT_MIN_K_PER_SPLIT = 64
DEFAULT_INPUT_SLOTS = 2
DEFAULT_COMPUTE_SLOTS = 2
DEFAULT_SWIZZLE_WINDOWS = ((1, 1), (2, 8), (4, 8))
DAV_2201_SWIZZLE_WINDOWS = ((1, 1), (2, 8), (4, 6))
DEFAULT_OUTPUT_PATHS = ("gm", "onchip_reduce")
DAV_3510_OUTPUT_PATHS = ("gm", "l0c_to_ub", "onchip_reduce")
DAV_2201_OUTPUT_PATHS = ("gm", "onchip_reduce")
DEFAULT_MICRO_TILES = {
    "bf16": ((128, 128, 128), (64, 64, 128), (16, 16, 16)),
    "fp16": ((128, 128, 128), (64, 64, 128), (16, 16, 16)),
    "fp32": ((128, 128, 64), (64, 64, 64), (16, 16, 16)),
    "int8": ((128, 128, 128), (64, 64, 128), (16, 16, 32)),
    "fp8": ((128, 128, 128), (64, 64, 128), (16, 16, 32)),
}
DAV_2201_MICRO_TILES = {
    key: value for key, value in DEFAULT_MICRO_TILES.items() if key != "fp8"
}
MECHANISM_SECTIONS = (
    "memory_hierarchy", "matrix", "attention", "indexer", "vector", "collective",
)
OUTPUT_PATH_ROUTES = {
    "gm": ("l0c", "gm"),
    "l0c_to_ub": ("l0c", "ub"),
}
ONCHIP_REDUCE_OUTPUT_PATH = "onchip_reduce"
LOCAL_ACCUMULATION_OUTPUT_PATHS = frozenset({ONCHIP_REDUCE_OUTPUT_PATH})
SHARED_L2_BUFFER_UNITS = 1


def _frozen_mapping(value) -> Mapping:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ModelingError("physical_mechanism_mapping_required")
    return dict(value)


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or int(value) < 1:
        raise ModelingError(f"physical_mechanism_positive_integer_required:{name}")
    return int(value)


def _fraction(value, name: str) -> float:
    parsed = float(value)
    if parsed <= 0 or parsed > 1:
        raise ModelingError(f"physical_mechanism_fraction_invalid:{name}")
    return parsed


def _pairs(value, name: str) -> tuple[tuple[int, int], ...]:
    result = []
    for item in value:
        if len(item) != PAIR_LEN:
            raise ModelingError(f"physical_mechanism_pair_required:{name}")
        result.append((_positive_int(item[0], name), _positive_int(item[1], name)))
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
                           max(0, int(self.l2_capacity_bytes)))
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
    mapped_refill_k_tiles: tuple[int, ...] = DEFAULT_MAPPED_REFILL_K_TILES
    body_candidates: tuple[tuple[int, int, int], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "base_tiles", _pairs(self.base_tiles, "base_tiles"))
        refills = tuple(_positive_int(value, "refill_k_tiles")
                        for value in self.refill_k_tiles)
        object.__setattr__(self, "refill_k_tiles", refills)
        micro = {str(key): _triples(value, f"micro_tiles:{key}")
                 for key, value in dict(self.micro_tiles_by_dtype).items()}
        if not micro:
            raise ModelingError("physical_mechanism_micro_tiles_empty")
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
        mapped = tuple(sorted({_positive_int(value, "mapped_refill_k_tiles")
                               for value in self.mapped_refill_k_tiles}))
        if not mapped:
            raise ModelingError("physical_mechanism_mapped_refill_tiles_empty")
        missing = sorted(set(mapped) - set(refills))
        if missing:
            raise ModelingError(
                f"physical_mechanism_mapped_refill_not_declared:{missing[0]}")
        object.__setattr__(self, "mapped_refill_k_tiles", mapped)
        object.__setattr__(self, "body_candidates",
                           _triples(self.body_candidates, "body_candidates")
                           if self.body_candidates else ())
        if 1 not in splits:
            raise ModelingError("physical_mechanism_split_k_requires_unity")

    def validate_capabilities(self, compute_arch=None, *, buffer_units=None) -> None:
        """Reject matrix schedules the declared compute architecture cannot host."""
        if compute_arch is not None and compute_arch.has_native_modes():
            for dtype in sorted(self.micro_tiles_by_dtype):
                if not compute_arch.supports_native("matrix", dtype):
                    raise ModelingError(f"matrix_policy_dtype_not_native:{dtype}")
        if compute_arch is not None:
            for path in self.supported_output_paths:
                route = OUTPUT_PATH_ROUTES.get(path)
                if route is None:
                    if path not in LOCAL_ACCUMULATION_OUTPUT_PATHS:
                        raise ModelingError(f"matrix_policy_output_path_unknown:{path}")
                    continue
                if compute_arch.path(route[0], route[1]) is None:
                    raise ModelingError(
                        "matrix_policy_output_path_unsupported:"
                        f"{path}:{route[0]}->{route[1]}"
                    )
        if buffer_units is not None and "l2" in dict(buffer_units):
            if int(dict(buffer_units)["l2"]) != SHARED_L2_BUFFER_UNITS:
                raise ModelingError("shared_l2_buffer_requires_one_die_owner")

    @property
    def signature(self):
        return (
            tuple(self.base_tiles), tuple(self.refill_k_tiles),
            tuple((key, tuple(value)) for key, value in
                  sorted(self.micro_tiles_by_dtype.items())),
            self.input_slots, self.compute_slots,
            tuple(self.split_k_candidates), self.min_k_per_split,
            tuple(self.swizzle_windows), tuple(self.supported_output_paths),
            tuple(self.mapped_refill_k_tiles), tuple(self.body_candidates),
        )

    def to_dict(self) -> dict:
        result = {
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
        if self.mapped_refill_k_tiles != DEFAULT_MAPPED_REFILL_K_TILES:
            result["mapped_refill_k_tiles"] = list(self.mapped_refill_k_tiles)
        if self.body_candidates:
            result["body_candidates"] = [list(value) for value in self.body_candidates]
        return result


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


def default_physical_mechanisms(*, l2_capacity_bytes: int = 0, compute_arch=None,
                                declared_micro_tiles=None) -> PhysicalMechanismSpec:
    """Conservative declaration resolved against the declared architecture."""
    preset = getattr(compute_arch, "preset", "") if compute_arch is not None else ""
    dav_2201 = preset == DAV_2201_PRESET
    has_l0c_ub = (
        compute_arch is not None
        and bool(compute_arch.paths)
        and compute_arch.path("l0c", "ub") is not None
    )
    base_tiles = DAV_2201_BASE_TILES if dav_2201 else DEFAULT_BASE_TILES
    refills = DAV_2201_REFILL_K_TILES if dav_2201 else DEFAULT_REFILL_K_TILES
    mapped = DAV_2201_MAPPED_REFILL_K_TILES if dav_2201 else DEFAULT_MAPPED_REFILL_K_TILES
    splits = DAV_2201_SPLIT_K_CANDIDATES if dav_2201 else DEFAULT_SPLIT_K_CANDIDATES
    swizzle = DAV_2201_SWIZZLE_WINDOWS if dav_2201 else DEFAULT_SWIZZLE_WINDOWS
    paths = (
        DAV_2201_OUTPUT_PATHS if dav_2201
        else (DAV_3510_OUTPUT_PATHS if has_l0c_ub else DEFAULT_OUTPUT_PATHS)
    )
    micros = (
        declared_micro_tiles if declared_micro_tiles is not None
        else (DAV_2201_MICRO_TILES if dav_2201 else DEFAULT_MICRO_TILES)
    )
    if (declared_micro_tiles is None and compute_arch is not None
            and compute_arch.has_native_modes()):
        micros = {
            dtype: tiles for dtype, tiles in micros.items()
            if compute_arch.supports_native("matrix", dtype)
        }
        if not micros:
            raise ModelingError("matrix_policy_no_native_matrix_dtypes")
    if compute_arch is not None:
        supported = []
        for path in paths:
            if path in LOCAL_ACCUMULATION_OUTPUT_PATHS:
                supported.append(path)
                continue
            route = OUTPUT_PATH_ROUTES.get(path)
            if route is not None and compute_arch.path(route[0], route[1]) is not None:
                supported.append(path)
        paths = tuple(supported) or (ONCHIP_REDUCE_OUTPUT_PATH,)
    body_candidates = DAV_3510_BODY_CANDIDATES if preset == DAV_3510_PRESET else ()
    return PhysicalMechanismSpec(
        MemoryHierarchySpec(
            l2_capacity_bytes=max(0, int(l2_capacity_bytes)),
            l2_effective_fraction=DEFAULT_L2_EFFECTIVE_FRACTION,
            cache_line_bytes=DEFAULT_CACHE_LINE_BYTES,
            write_allocate=True,
            provenance={"basis": "legacy_profile_default", "calibrated": False},
        ),
        MatrixPolicySpec(
            base_tiles=base_tiles,
            refill_k_tiles=refills,
            micro_tiles_by_dtype=micros,
            input_slots=DEFAULT_INPUT_SLOTS,
            compute_slots=DEFAULT_COMPUTE_SLOTS,
            split_k_candidates=splits,
            min_k_per_split=DEFAULT_MIN_K_PER_SPLIT,
            swizzle_windows=swizzle,
            supported_output_paths=paths,
            mapped_refill_k_tiles=mapped,
            body_candidates=body_candidates,
        ),
    )


def _declared(matrix, name: str, default):
    value = matrix.get(name, default)
    if value is None:
        raise ModelingError(f"physical_mechanism_field_requires_value:{name}")
    return value


def parse_physical_mechanisms(value, *, fallback_l2_bytes: int = 0,
                              compute_arch=None, buffer_units=None
                              ) -> PhysicalMechanismSpec:
    """Parse one device declaration, retaining strict typed validation."""
    if value is None:
        defaults = default_physical_mechanisms(
            l2_capacity_bytes=fallback_l2_bytes, compute_arch=compute_arch)
        defaults.matrix.validate_capabilities(
            compute_arch, buffer_units=buffer_units)
        return defaults
    if not isinstance(value, Mapping):
        raise ModelingError("device_physical_mechanisms_requires_mapping")
    if any(section in value and value[section] is None
           for section in MECHANISM_SECTIONS):
        raise ModelingError("device_physical_mechanisms_sections_require_mapping")
    memory = value.get("memory_hierarchy", {})
    matrix = value.get("matrix", {})
    if not isinstance(memory, Mapping) or not isinstance(matrix, Mapping):
        raise ModelingError("device_physical_mechanisms_sections_require_mapping")
    declared_micros = (matrix.get("micro_tiles_by_dtype")
                       if "micro_tiles_by_dtype" in matrix else None)
    defaults = default_physical_mechanisms(
        l2_capacity_bytes=fallback_l2_bytes, compute_arch=compute_arch,
        declared_micro_tiles=declared_micros)
    hierarchy = MemoryHierarchySpec(
        l2_capacity_bytes=int(memory.get(
            "l2_capacity_bytes", defaults.memory_hierarchy.l2_capacity_bytes)),
        l2_effective_fraction=float(memory.get(
            "l2_effective_fraction",
            defaults.memory_hierarchy.l2_effective_fraction)),
        cache_line_bytes=int(memory.get(
            "cache_line_bytes", defaults.memory_hierarchy.cache_line_bytes)),
        write_allocate=bool(memory.get(
            "write_allocate", defaults.memory_hierarchy.write_allocate)),
        provenance=memory.get("provenance", defaults.memory_hierarchy.provenance),
    )
    policy = MatrixPolicySpec(
        base_tiles=_declared(matrix, "base_tiles", defaults.matrix.base_tiles),
        refill_k_tiles=_declared(matrix, "refill_k_tiles",
                                 defaults.matrix.refill_k_tiles),
        micro_tiles_by_dtype=_declared(matrix, "micro_tiles_by_dtype",
                                       defaults.matrix.micro_tiles_by_dtype),
        input_slots=int(_declared(matrix, "input_slots",
                                  defaults.matrix.input_slots)),
        compute_slots=int(_declared(matrix, "compute_slots",
                                    defaults.matrix.compute_slots)),
        split_k_candidates=_declared(matrix, "split_k_candidates",
                                     defaults.matrix.split_k_candidates),
        min_k_per_split=int(_declared(matrix, "min_k_per_split",
                                      defaults.matrix.min_k_per_split)),
        swizzle_windows=_declared(matrix, "swizzle_windows",
                                  defaults.matrix.swizzle_windows),
        supported_output_paths=_declared(matrix, "supported_output_paths",
                                         defaults.matrix.supported_output_paths),
        mapped_refill_k_tiles=_declared(matrix, "mapped_refill_k_tiles",
                                        defaults.matrix.mapped_refill_k_tiles),
        body_candidates=(
            tuple(matrix.get("body_candidates") or ())
            if "body_candidates" in matrix
            else defaults.matrix.body_candidates
        ),
    )
    policy.validate_capabilities(compute_arch, buffer_units=buffer_units)
    return PhysicalMechanismSpec(
        hierarchy, policy,
        attention=value.get("attention") or {},
        indexer=value.get("indexer") or {},
        vector=value.get("vector") or {},
        collective=value.get("collective") or {},
    )
