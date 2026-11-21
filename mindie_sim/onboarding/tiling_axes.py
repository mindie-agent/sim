"""Declarative TilingAxes contracts and ABI guard validation (Prevention Lock 3).

[Role] Define immutable ``AbiDivisibilityRule`` and ``TilingAxes`` specifications
  plus deterministic ``validate_abi`` filtering so onboarding cliff planners
  automatically synthesize 2D ``(B/q x Context)`` boundary suites and route
  illegal ABI shapes (such as ``topk=2049`` when ``topk > 2048`` requires
  divisibility by ``1024``) into ABI-rejection tests instead of performance
  benchmarks.
[Invariants] Pure structural validation; zero SKU-name sniffing, zero numeric
  magic literals, and deterministic rejection reasons.
[Timeouts & Locks] In-memory dataclass construction and O(rules) validation;
  zero I/O, locks, or device contexts.
[Side-Effect Policy] Zero side effects on import or invocation.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from mindie_sim.schema.common import ModelingError

CROSS_SAMPLE_BATCH_SIZES: tuple[int, ...] = (16, 64, 128)
CROSS_SAMPLE_MTP_Q: tuple[int, ...] = (2, 4, 8)
CROSS_SAMPLE_CHUNK_Q: tuple[int, ...] = (2048, 8192)
CROSS_SAMPLE_PARALLEL_CONTEXT_MIN = 131072
LIGHTNING_TOPK_ABOVE_THRESHOLD = 2048
LIGHTNING_TOPK_DIVISOR = 1024
DEFAULT_TILING_BATCH_AXIS: tuple[int, ...] = (1, 16, 64, 128)
DEFAULT_TILING_QUERY_AXIS: tuple[int, ...] = (1, 2, 4, 8, 2048, 8192)


@dataclass(frozen=True, slots=True)
class AbiDivisibilityRule:
    """Upstream ABI divisibility and range guard for one operator shape axis."""

    axis: str
    above_threshold: int
    divisor: int
    min_value: int = 1
    max_value: int | None = None
    rationale: str = ""

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

    def check(self, value: int) -> tuple[bool, str]:
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
        return asdict(self)


def default_topk_abi_rule(
    *,
    above_threshold: int = LIGHTNING_TOPK_ABOVE_THRESHOLD,
    divisor: int = LIGHTNING_TOPK_DIVISOR,
) -> AbiDivisibilityRule:
    """Canonical sparse-indexer topk ABI guard (``topk > 2048`` divisible by ``1024``)."""
    return AbiDivisibilityRule(
        axis="topk",
        above_threshold=int(above_threshold),
        divisor=int(divisor),
        min_value=1,
        rationale="upstream_lightning_indexer_v2_topk_above_2048_requires_1024_alignment",
    )


@dataclass(frozen=True, slots=True)
class TilingAxes:
    """Declarative tiling, core-product, phase-threshold, and ABI guard spec."""

    tile_dims: tuple[int, ...]
    core_product_axes: tuple[str, ...] = ("tokens", "batch_x_kv_heads")
    phase_thresholds: tuple[tuple[str, int], ...] = ()
    abi_divisibility: tuple[AbiDivisibilityRule, ...] = ()
    batch_axis: tuple[int, ...] = DEFAULT_TILING_BATCH_AXIS
    query_tokens_axis: tuple[int, ...] = DEFAULT_TILING_QUERY_AXIS

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
        }


def validate_abi(
    point: Mapping[str, Any],
    axes: TilingAxes,
) -> tuple[bool, str]:
    """Validate one candidate benchmark shape against ``axes.abi_divisibility``."""
    axis_name = str(point.get("axis") or "")
    axis_val = point.get("value")
    resolved_values: dict[str, int] = {}
    if axis_name and type(axis_val) is int:
        resolved_values[axis_name] = axis_val
    for key in ("context_len", "topk", "batch_size", "query_tokens", "tokens"):
        val = point.get(key)
        if val is not None and type(val) is int:
            resolved_values[key] = val

    for rule in axes.abi_divisibility:
        if rule.axis in resolved_values:
            ok, reason = rule.check(resolved_values[rule.axis])
            if not ok:
                return False, reason

    if "topk" in resolved_values and "context_len" in resolved_values:
        if resolved_values["topk"] > resolved_values["context_len"]:
            return (
                False,
                f"abi_topk_exceeds_context:topk={resolved_values['topk']}"
                f">context={resolved_values['context_len']}",
            )
    return True, ""


__all__ = [
    "CROSS_SAMPLE_BATCH_SIZES",
    "CROSS_SAMPLE_CHUNK_Q",
    "CROSS_SAMPLE_MTP_Q",
    "CROSS_SAMPLE_PARALLEL_CONTEXT_MIN",
    "DEFAULT_TILING_BATCH_AXIS",
    "DEFAULT_TILING_QUERY_AXIS",
    "LIGHTNING_TOPK_ABOVE_THRESHOLD",
    "LIGHTNING_TOPK_DIVISOR",
    "AbiDivisibilityRule",
    "TilingAxes",
    "default_topk_abi_rule",
    "validate_abi",
]
