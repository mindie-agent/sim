"""Declarative TilingAxes contracts and ISA-dispatched ABI validation (Lock 3).

[Role] Define immutable ``AbiDivisibilityRule`` and ``TilingAxes`` specs plus
  deterministic ISA-dispatched ``validate_abi`` and ``family_default_tiling_axes``
  loaders so onboarding cliff planners automatically load target family rules,
  enforce integer ``1..8192`` topk bounds, dispatch Arch22 vs Arch35 ABI rules
  faithfully (preserving upstream-valid Arch35 ``context=1, topk=64``), and
  route illegal ABI shapes into ABI-rejection tests.
[Invariants] Pure structural validation; rejects non-int coordinates (strings,
  bools, floats) with ``abi_type_violation``; zero SKU-name sniffing; zero
  numeric magic literals; deterministic rejection reasons.
[Timeouts & Locks] In-memory dataclass construction and O(rules) validation;
  zero I/O, locks, or device contexts.
[Side-Effect Policy] Zero side effects on import or invocation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
import importlib
from typing import Any

from mindie_sim.schema.common import ModelingError

ARCH_PRESET_22 = "npu_arch_2201"
ARCH_PRESET_35 = "npu_arch_3510"
CROSS_SAMPLE_BATCH_SIZES: tuple[int, ...] = (16, 64, 128)
CROSS_SAMPLE_MTP_Q: tuple[int, ...] = (2, 4, 8)
CROSS_SAMPLE_CHUNK_Q: tuple[int, ...] = (2048, 8192)
CROSS_SAMPLE_PARALLEL_CONTEXT_MIN = 131072
LIGHTNING_TOPK_ABOVE_THRESHOLD = 2048
LIGHTNING_TOPK_DIVISOR = 1024
LIGHTNING_TOPK_MAX_VALUE = 8192
QLI_SMALL_M_TOPK_THRESHOLD = 6144
DEFAULT_TILING_BATCH_AXIS: tuple[int, ...] = (1, 16, 64, 128)
DEFAULT_TILING_QUERY_AXIS: tuple[int, ...] = (1, 2, 4, 8, 2048, 8192)
_SPARSE_INDEXER_FAMILIES = frozenset({"sparse_indexer", "lightning_indexer", "indexer"})
_CORE_COORD_KEYS = ("context_len", "topk", "batch_size", "query_tokens", "tokens")


@dataclass(frozen=True, slots=True)
class AbiDivisibilityRule:
    """Upstream ABI divisibility, range, and ISA applicability guard."""

    axis: str
    above_threshold: int
    divisor: int
    min_value: int = 1
    max_value: int | None = None
    rationale: str = ""
    arch_presets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.axis.strip():
            raise ModelingError("abi_rule_axis_empty")
        if type(self.above_threshold) is not int or self.above_threshold < 0:
            raise ModelingError(f"abi_rule_threshold_invalid:{self.above_threshold}")
        if type(self.divisor) is not int or self.divisor < 1:
            raise ModelingError(f"abi_rule_divisor_invalid:{self.divisor}")
        if type(self.min_value) is not int or self.min_value < 0:
            raise ModelingError(f"abi_rule_min_invalid:{self.min_value}")
        if self.max_value is not None and (
            type(self.max_value) is not int or self.max_value < self.min_value
        ):
            raise ModelingError(f"abi_rule_max_invalid:{self.max_value}")

    def applies_to(self, arch_preset: str) -> bool:
        """Return True iff this rule applies to ``arch_preset`` (empty = all ISAs)."""
        if not self.arch_presets or not arch_preset:
            return True
        return arch_preset in self.arch_presets

    def check(self, value: Any) -> tuple[bool, str]:
        """Return ``(True, '')`` when ``value`` obeys the ABI contract, else rejection reason."""
        if type(value) is not int:
            return False, f"abi_type_violation:{self.axis}={value!r}"
        if value < self.min_value:
            return False, f"abi_min_violation:{self.axis}={value}<min={self.min_value}"
        if self.max_value is not None and value > self.max_value:
            return False, f"abi_max_violation:{self.axis}={value}>max={self.max_value}"
        if value > self.above_threshold and value % self.divisor != 0:
            return (
                False,
                f"abi_divisibility_violation:{self.axis}={value}"
                f":above={self.above_threshold}:divisor={self.divisor}",
            )
        return True, ""

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["arch_presets"] = list(self.arch_presets)
        return raw


def default_topk_abi_rule(
    *,
    above_threshold: int = LIGHTNING_TOPK_ABOVE_THRESHOLD,
    divisor: int = LIGHTNING_TOPK_DIVISOR,
    max_value: int = LIGHTNING_TOPK_MAX_VALUE,
    arch_presets: tuple[str, ...] = (),
) -> AbiDivisibilityRule:
    """Canonical sparse-indexer topk ABI guard (``1..8192``; ``>2048`` divisible by ``1024``)."""
    return AbiDivisibilityRule(
        axis="topk",
        above_threshold=int(above_threshold),
        divisor=int(divisor),
        min_value=1,
        max_value=int(max_value),
        rationale="upstream_lightning_indexer_v2_topk_1_to_8192_above_2048_requires_1024_alignment",
        arch_presets=tuple(arch_presets),
    )


@dataclass(frozen=True, slots=True)
class TilingAxes:
    """Declarative tiling, core-product, phase-threshold, and ISA ABI guard spec."""

    tile_dims: tuple[int, ...]
    core_product_axes: tuple[str, ...] = ("tokens", "batch_x_kv_heads")
    phase_thresholds: tuple[tuple[str, int], ...] = ()
    abi_divisibility: tuple[AbiDivisibilityRule, ...] = ()
    batch_axis: tuple[int, ...] = DEFAULT_TILING_BATCH_AXIS
    query_tokens_axis: tuple[int, ...] = DEFAULT_TILING_QUERY_AXIS
    enforce_topk_le_context_archs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.tile_dims:
            raise ModelingError("tiling_axes_empty_tile_dims")
        for dim in self.tile_dims:
            if type(dim) is not int or dim < 2:
                raise ModelingError(f"tiling_axes_tile_dim_invalid:{dim}")
        for axis, threshold in self.phase_thresholds:
            if not str(axis).strip() or type(threshold) is not int or threshold < 2:
                raise ModelingError(f"tiling_axes_threshold_invalid:{axis}:{threshold}")
        for batch in self.batch_axis:
            if type(batch) is not int or batch < 1:
                raise ModelingError(f"tiling_axes_batch_invalid:{batch}")
        for q_tok in self.query_tokens_axis:
            if type(q_tok) is not int or q_tok < 1:
                raise ModelingError(f"tiling_axes_query_invalid:{q_tok}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tile_dims": list(self.tile_dims),
            "core_product_axes": list(self.core_product_axes),
            "phase_thresholds": {axis: value for axis, value in self.phase_thresholds},
            "abi_divisibility": [rule.to_dict() for rule in self.abi_divisibility],
            "batch_axis": list(self.batch_axis),
            "query_tokens_axis": list(self.query_tokens_axis),
            "enforce_topk_le_context_archs": list(self.enforce_topk_le_context_archs),
        }


def family_default_tiling_axes(
    family: str,
    *,
    tile_dim: int,
    arch_preset: str = "",
    topk_threshold: int | None = None,
) -> TilingAxes:
    """Load declared family ``TILING_AXES`` or synthesize ISA-dispatched family defaults."""
    clean_family = str(family or "").strip()
    for mod_path in (
        f"mindie_sim.prediction.operators.{clean_family}.common",
        f"mindie_sim.prediction.operators.{clean_family}",
    ):
        if not clean_family:
            break
        try:
            mod = importlib.import_module(mod_path)
        except Exception:
            continue
        declared = getattr(mod, "TILING_AXES", None)
        if callable(declared):
            resolved = declared(arch_preset)
            if isinstance(resolved, TilingAxes):
                return resolved
        elif isinstance(declared, TilingAxes):
            return declared

    if clean_family in _SPARSE_INDEXER_FAMILIES:
        tk = int(topk_threshold) if topk_threshold is not None else LIGHTNING_TOPK_ABOVE_THRESHOLD
        bounds_rule = default_topk_abi_rule(
            above_threshold=LIGHTNING_TOPK_ABOVE_THRESHOLD,
            divisor=1,
            max_value=LIGHTNING_TOPK_MAX_VALUE,
            arch_presets=(),
        )
        dav22_div_rule = default_topk_abi_rule(
            above_threshold=LIGHTNING_TOPK_ABOVE_THRESHOLD,
            divisor=LIGHTNING_TOPK_DIVISOR,
            max_value=LIGHTNING_TOPK_MAX_VALUE,
            arch_presets=(ARCH_PRESET_22,),
        )
        if arch_preset == ARCH_PRESET_35:
            thresholds: tuple[tuple[str, int], ...] = (
                (("topk", tk), ("topk", QLI_SMALL_M_TOPK_THRESHOLD))
                if tk != QLI_SMALL_M_TOPK_THRESHOLD
                else (("topk", tk),)
            )
            return TilingAxes(
                tile_dims=(int(tile_dim),),
                core_product_axes=("tokens", "batch_x_kv_heads"),
                phase_thresholds=thresholds,
                abi_divisibility=(bounds_rule, dav22_div_rule),
                enforce_topk_le_context_archs=(),
            )
        return TilingAxes(
            tile_dims=(int(tile_dim),),
            core_product_axes=("tokens", "batch_x_kv_heads"),
            phase_thresholds=(("topk", tk),),
            abi_divisibility=(bounds_rule, dav22_div_rule),
            enforce_topk_le_context_archs=(),
        )

    thresholds_list: list[tuple[str, int]] = []
    rules: list[AbiDivisibilityRule] = []
    if topk_threshold is not None:
        tk_val = int(topk_threshold)
        thresholds_list.append(("topk", tk_val))
        rules.append(
            default_topk_abi_rule(
                above_threshold=LIGHTNING_TOPK_ABOVE_THRESHOLD,
                divisor=1,
                max_value=LIGHTNING_TOPK_MAX_VALUE,
                arch_presets=(),
            )
        )
        rules.append(
            default_topk_abi_rule(
                above_threshold=LIGHTNING_TOPK_ABOVE_THRESHOLD,
                divisor=LIGHTNING_TOPK_DIVISOR,
                max_value=LIGHTNING_TOPK_MAX_VALUE,
                arch_presets=(ARCH_PRESET_22,),
            )
        )
    return TilingAxes(
        tile_dims=(int(tile_dim),),
        core_product_axes=("tokens", "batch_x_kv_heads"),
        phase_thresholds=tuple(thresholds_list),
        abi_divisibility=tuple(rules),
    )


def validate_abi(
    point: Mapping[str, Any],
    axes: TilingAxes,
    *,
    arch_preset: str = "",
) -> tuple[bool, str]:
    """Validate one candidate benchmark shape against ISA-applicable ``axes`` ABI rules."""
    axis_name = str(point.get("axis") or "").strip()
    axis_val = point.get("value")
    resolved_values: dict[str, int] = {}
    if axis_name and axis_val is not None:
        if type(axis_val) is not int:
            return False, f"abi_type_violation:{axis_name}={axis_val!r}"
        resolved_values[axis_name] = axis_val

    rule_axes = tuple(dict.fromkeys((*_CORE_COORD_KEYS, *(r.axis for r in axes.abi_divisibility))))
    for key in rule_axes:
        if key not in point or point[key] is None:
            continue
        val = point[key]
        if type(val) is not int:
            return False, f"abi_type_violation:{key}={val!r}"
        resolved_values[key] = val

    for rule in axes.abi_divisibility:
        if not rule.applies_to(arch_preset):
            continue
        if rule.axis in resolved_values:
            ok, reason = rule.check(resolved_values[rule.axis])
            if not ok:
                return False, reason

    if (
        arch_preset
        and arch_preset in axes.enforce_topk_le_context_archs
        and "topk" in resolved_values
        and "context_len" in resolved_values
        and resolved_values["topk"] > resolved_values["context_len"]
    ):
        return (
            False,
            f"abi_topk_exceeds_context:topk={resolved_values['topk']}"
            f">context={resolved_values['context_len']}",
        )
    return True, ""


__all__ = [
    "ARCH_PRESET_22",
    "ARCH_PRESET_35",
    "CROSS_SAMPLE_BATCH_SIZES",
    "CROSS_SAMPLE_CHUNK_Q",
    "CROSS_SAMPLE_MTP_Q",
    "CROSS_SAMPLE_PARALLEL_CONTEXT_MIN",
    "DEFAULT_TILING_BATCH_AXIS",
    "DEFAULT_TILING_QUERY_AXIS",
    "LIGHTNING_TOPK_ABOVE_THRESHOLD",
    "LIGHTNING_TOPK_DIVISOR",
    "LIGHTNING_TOPK_MAX_VALUE",
    "QLI_SMALL_M_TOPK_THRESHOLD",
    "AbiDivisibilityRule",
    "TilingAxes",
    "default_topk_abi_rule",
    "family_default_tiling_axes",
    "validate_abi",
]
