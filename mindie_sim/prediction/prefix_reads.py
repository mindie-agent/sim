"""Physical prefix read reuse for batch index and attention access.

[Role]
  Separate logical per-query key traffic from unique physical HBM fills.  Shared
  storage alone does not imply a source kernel reads a prefix once, so reuse is
  applied only by a declared batch-reuse mechanism; a source program keeps its
  own per-sequence semantics.

[Invariants]
  * HBM fill bytes deduplicate identical immutable physical ranges within the
    declared reuse scope; MTE2/core bytes are reported separately because a
    resident tile can feed several core transfers.
  * Score FLOPs, query reads, writes, TopK work and output bytes never shrink.
  * Prefix ownership and compression determine the prefix extent in index-row
    space; speculative/private rows remain request-specific.
  * ``per_sequence`` preserves the old read traffic even with pooled capacity.

[Timeouts & Locks]
  Bounded by accesses x ranges; no I/O or locks.

[Side-Effect Policy]
  Pure immutable records.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from typing import Any

from mindie_sim.schema.common import ModelingError


FIRST = 0
ZERO_FLOAT = 0.0
ONE_FLOAT = 1.0
SAVINGS_EPS = 1e-9
PAIR = 2
MODE_AUTO = "auto"
MODE_PER_SEQUENCE = "per_sequence"
MODE_BATCH_SHARED = "batch_shared"
MODES = frozenset({MODE_AUTO, MODE_PER_SEQUENCE, MODE_BATCH_SHARED})
REUSE_TILE = "tile"
REUSE_PAGED_L2 = "paged_l2"
REUSE_PER_SEQUENCE = "per_sequence"
SUPPORTED_BATCH_SHARED_REUSE = frozenset({REUSE_TILE, "batch_tile"})
SUPPORTED_AUTO_REUSE = frozenset({REUSE_TILE, "batch_tile", REUSE_PAGED_L2})
NO_REUSE_REASONS = {
    "mode_not_shared": "prefix_read_mode_not_batch_shared",
    "single_access": "prefix_read_single_access",
    "empty_prefix": "prefix_read_empty_shared_prefix",
    "unsupported_kernel": "prefix_read_kernel_reuse_unsupported",
    "no_savings": "prefix_read_no_shared_savings",
}


@dataclass(frozen=True)
class CacheAccess:
    """One query's declared key-cache access geometry."""

    request_id: str
    context_tokens: int
    q_len: int = 1
    prefix_tokens: int = 0
    prefix_key: str = ""
    row_bytes: float = ONE_FLOAT
    scale_bytes: float = ZERO_FLOAT
    layer: int = 0
    component: str = "index_k"
    selected_tokens: tuple[tuple[int, int], ...] = ()
    logical_rows: int = 0
    unique_rows: int = 0
    window_tokens: int | None = None
    shared_intervals: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class PrefixReadPlan:
    """Logical versus unique-physical prefix read accounting."""

    logical_query_bytes: float
    unique_hbm_bytes: float
    l2_to_core_bytes: float
    query_read_bytes: float
    scale_read_bytes: float
    write_bytes: float
    workspace_bytes: float
    reuse_mechanism: str
    read_reuse_applied: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_query_bytes": float(self.logical_query_bytes),
            "unique_hbm_bytes": float(self.unique_hbm_bytes),
            "l2_to_core_bytes": float(self.l2_to_core_bytes),
            "query_read_bytes": float(self.query_read_bytes),
            "scale_read_bytes": float(self.scale_read_bytes),
            "write_bytes": float(self.write_bytes),
            "workspace_bytes": float(self.workspace_bytes),
            "reuse_mechanism": self.reuse_mechanism,
            "read_reuse_applied": bool(self.read_reuse_applied),
            "reason": self.reason,
        }



def _logical_rows(access: CacheAccess) -> float:
    if int(access.logical_rows) > 0:
        return float(int(access.logical_rows))
    context = max(0, int(access.context_tokens))
    q = max(1, int(access.q_len))
    total = 0
    for index in range(q):
        total += min(context + index + 1, context + q)
    return float(max(1, total))


def _merged_intervals(intervals) -> tuple[tuple[int, int], ...]:
    """Canonical disjoint half-open intervals sorted by start."""
    ordered = sorted((int(start), int(end)) for start, end in intervals
                     if int(end) > int(start))
    merged: list[tuple[int, int]] = []
    for first, last in ordered:
        if not merged or first > merged[-1][1]:
            merged.append((first, last))
        elif last > merged[-1][1]:
            merged[-1] = (merged[-1][0], last)
    return tuple(merged)


def _merged_interval_rows(intervals) -> int:
    """Union length of absolute half-open row intervals."""
    return sum(end - start for start, end in _merged_intervals(intervals))


def owned_window_intervals(
    intervals, cp: int, interleave: int, rank: int,
) -> tuple[tuple[int, int], ...]:
    """Rank-owned sub-intervals under interleaved DCP token ownership."""
    merged = _merged_intervals(intervals)
    if cp <= 1 or not merged:
        return merged
    step = max(1, int(interleave))
    owned: list[tuple[int, int]] = []
    for lo, hi in merged:
        first_block = lo // step
        block = first_block + ((int(rank) - (first_block % cp)) % cp)
        while block * step < hi:
            seg_lo = max(lo, block * step)
            seg_hi = min(hi, (block + 1) * step)
            if seg_hi > seg_lo:
                owned.append((seg_lo, seg_hi))
            block += cp
    return _merged_intervals(owned)


def _materialized_prefix(access: CacheAccess, cache_state: Mapping[str, Any] | None) -> int:
    declared = max(0, int(access.prefix_tokens))
    if not cache_state:
        return declared
    entry = cache_state.get(access.prefix_key, cache_state.get("*"))
    if entry is None:
        return 0
    if isinstance(entry, Mapping):
        limit = entry.get("materialized_tokens", entry.get("valid_tokens"))
        if limit is None:
            return declared
        return max(0, min(declared, int(limit)))
    if isinstance(entry, (int, float)) and not isinstance(entry, bool):
        return max(0, min(declared, int(entry)))
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        spans = tuple(entry)
        if spans and all(isinstance(item, Sequence) and len(item) == PAIR for item in spans):
            return max((int(end) for _start, end in spans), default=0)
    return declared


def _intersect_intervals(
    intervals: Sequence[tuple[int, int]], lo: int, hi: int
) -> tuple[tuple[int, int], ...]:
    if hi <= lo:
        return ()
    clipped = [
        (max(lo, int(start)), min(hi, int(end)))
        for start, end in intervals
        if min(hi, int(end)) > max(lo, int(start))
    ]
    return _merged_intervals(clipped)


def _l2_capacity_bytes(geometry: Any) -> float | None:
    if geometry is None:
        return None
    if isinstance(geometry, Mapping):
        raw = geometry.get("effective_l2_bytes", geometry.get("l2_capacity_bytes"))
    else:
        raw = getattr(
            geometry, "effective_l2_bytes", getattr(geometry, "l2_capacity_bytes", None)
        )
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and float(raw) > 0.0:
        return float(raw)
    return None


def plan_prefix_reads(
    accesses: Sequence[CacheAccess], *, mode: str, kernel_reuse: str,
    geometry: Any = None, cache_state: Mapping[str, Any] | None = None,
) -> PrefixReadPlan:
    """Project logical key reads onto unique physical ranges."""
    if mode not in MODES:
        raise ModelingError(f"prefix_read_mode_unsupported:{mode}")
    accesses = tuple(accesses)
    logical = math.fsum(
        float(_logical_rows(access)) * float(access.row_bytes)
        + float(access.scale_bytes) * max(1, int(access.q_len))
        for access in accesses)
    query_reads = math.fsum(
        float(max(1, int(access.q_len))) * float(access.row_bytes)
        for access in accesses)
    scale_total = math.fsum(
        float(access.scale_bytes) * max(1, int(access.q_len))
        for access in accesses)
    if not accesses or mode == MODE_PER_SEQUENCE:
        return PrefixReadPlan(
            logical, logical, logical, query_reads, scale_total, 0, 0,
            REUSE_PER_SEQUENCE, False, NO_REUSE_REASONS["mode_not_shared"])
    shared_candidates = [access for access in accesses if int(access.prefix_tokens) > 0]
    if len(accesses) <= 1:
        return PrefixReadPlan(
            logical, logical, logical, query_reads, scale_total, 0, 0,
            REUSE_PER_SEQUENCE, False, NO_REUSE_REASONS["single_access"])
    if mode == MODE_BATCH_SHARED and kernel_reuse not in SUPPORTED_BATCH_SHARED_REUSE:
        raise ModelingError("shared_prefix_kernel_reuse_unsupported")
    if mode == MODE_AUTO and kernel_reuse not in SUPPORTED_AUTO_REUSE:
        return PrefixReadPlan(
            logical, logical, logical, query_reads, scale_total, 0, 0,
            REUSE_PER_SEQUENCE, False, NO_REUSE_REASONS["unsupported_kernel"])
    if not shared_candidates:
        return PrefixReadPlan(
            logical, logical, logical, query_reads, scale_total, 0, 0,
            REUSE_PER_SEQUENCE, False, NO_REUSE_REASONS["empty_prefix"])
    unique = 0
    core = 0
    logical_rows = math.fsum(_logical_rows(access) for access in accesses)
    unique_rows = 0
    logical_shared_bytes = ZERO_FLOAT
    groups: dict[tuple[int, str, str, float], list[tuple[int, int]]] = {}
    for access in accesses:
        context = max(0, int(access.context_tokens))
        q_len = max(1, int(access.q_len))
        rows = _logical_rows(access)
        prefix_limit = _materialized_prefix(access, cache_state)
        end = context + q_len
        if access.window_tokens is not None:
            win = max(0, int(access.window_tokens))
            windowed = win > 0
            start = max(0, end - win) if windowed else 0
        else:
            windowed = rows < end and int(prefix_limit) < int(rows)
            start = max(0, end - int(rows)) if windowed else 0
        span = (int(access.unique_rows)
                if int(access.unique_rows) > 0
                else (int(rows) if not windowed and int(access.logical_rows) > 0
                      else max(0, end - start)))
        if access.shared_intervals:
            shared = min(_merged_interval_rows(access.shared_intervals), span)
            private = max(0, span - shared)
            intervals = access.shared_intervals
        elif access.selected_tokens:
            visible_sel = _intersect_intervals(access.selected_tokens, start, end)
            shared_iv = _intersect_intervals(visible_sel, 0, max(0, int(prefix_limit)))
            priv_iv = _intersect_intervals(visible_sel, max(start, int(prefix_limit)), end)
            shared = _merged_interval_rows(shared_iv)
            private = _merged_interval_rows(priv_iv)
            intervals = shared_iv
        elif windowed:
            shared_lo = start
            shared_hi = min(end, max(0, int(prefix_limit)))
            raw_shared = max(0, shared_hi - shared_lo)
            shared = min(raw_shared, span)
            private = max(0, span - shared)
            intervals = ((shared_lo, shared_lo + shared),) if shared > 0 else ()
        else:
            limit = int(access.logical_rows) if int(access.logical_rows) > 0 else context
            prefix = min(max(0, int(prefix_limit)), limit)
            shared = min(prefix, span)
            private = max(0, span - shared)
            intervals = ((0, shared),) if shared > 0 else ()
        unique += float(private) * float(access.row_bytes)
        unique_rows += float(private)
        logical_shared_bytes += float(shared) * float(access.row_bytes)
        core += rows * float(access.row_bytes)
        if shared > 0 and intervals:
            key = (int(access.layer), str(access.component),
                   str(access.prefix_key), float(access.row_bytes))
            groups.setdefault(key, []).extend(intervals)
    unique_shared_bytes = ZERO_FLOAT
    for key, intervals in groups.items():
        shared_rows = _merged_interval_rows(intervals)
        shared_b = float(shared_rows) * float(key[3])
        unique += shared_b
        unique_shared_bytes += shared_b
        unique_rows += float(shared_rows)
    l2_cap = _l2_capacity_bytes(geometry)
    if kernel_reuse == REUSE_PAGED_L2 and l2_cap is not None and unique_shared_bytes > ZERO_FLOAT:
        retained = min(unique_shared_bytes, l2_cap)
        spill_ratio = max(ZERO_FLOAT, ONE_FLOAT - retained / unique_shared_bytes)
        refill_shared = max(ZERO_FLOAT, logical_shared_bytes - unique_shared_bytes) * spill_ratio
        unique += refill_shared
    if logical_rows > 0:
        unique_scale = scale_total * unique_rows / logical_rows
    else:
        unique_scale = scale_total
    unique = min(logical, unique + unique_scale)
    core += scale_total
    applied = bool((logical - unique) > SAVINGS_EPS)
    return PrefixReadPlan(
        logical, unique, core, query_reads, scale_total, 0, 0,
        kernel_reuse if applied else REUSE_PER_SEQUENCE, applied,
        "" if applied else NO_REUSE_REASONS["no_savings"])


def apply_prefix_read_plan(op: Any, plan: PrefixReadPlan, *,
                           hbm_component: float | None = None,
                           mte2_component: float | None = None) -> Any:
    """Replace one precharged key-read component with its physical union.

    ``HBM_new = HBM_old - KV_HBM_precharged + KV_HBM_union`` and
    ``MTE2_new = MTE2_old - KV_MTE2_precharged + KV_core_refills``.  Both
    components are supplied by the caller that constructed them, so query,
    output and score traffic outside the key component is never subtracted and
    no ``max(0, ...)`` repair can erase mandatory work.
    """
    metadata = dict(getattr(op, "metadata", ()) or ())
    metadata.update({
        "prefix_read_mode": plan.reuse_mechanism,
        "prefix_read_reuse_applied": bool(plan.read_reuse_applied),
        "prefix_read_logical_bytes": float(plan.logical_query_bytes),
        "prefix_read_hbm_bytes": float(plan.unique_hbm_bytes),
        "prefix_read_l2_core_bytes": float(plan.l2_to_core_bytes),
        "prefix_read_reason": plan.reason,
    })
    hbm = float(getattr(op, "hbm_bytes", 0))
    mte2 = float(getattr(op, "mte2_bytes", 0))
    if plan.read_reuse_applied:
        # A caller that does not name its key component keeps the historical
        # delta form; the component-aware callers pass the exact precharged
        # bytes so query/output/score payloads are never touched.
        hbm_precharged = (float(plan.logical_query_bytes)
                          if hbm_component is None else float(hbm_component))
        mte2_precharged = (float(plan.logical_query_bytes)
                           if mte2_component is None else float(mte2_component))
        if ((hbm_component is not None and
             not ZERO_FLOAT <= hbm_precharged <= hbm + SAVINGS_EPS)
                or (mte2_component is not None and
                    not ZERO_FLOAT <= mte2_precharged <= mte2 + SAVINGS_EPS)):
            raise ModelingError("prefix_read_component_exceeds_declared_work")
        hbm = hbm - hbm_precharged + float(plan.unique_hbm_bytes)
        mte2 = mte2 - mte2_precharged + float(plan.l2_to_core_bytes)
        if hbm < -SAVINGS_EPS or mte2 < -SAVINGS_EPS:
            raise ModelingError(
                "prefix_read_component_underflow:"
                f"{hbm_precharged:.3f}>"
                f"{float(getattr(op, 'hbm_bytes', 0)):.3f}:"
                f"{mte2_precharged:.3f}>"
                f"{float(getattr(op, 'mte2_bytes', 0)):.3f}")
        hbm, mte2 = max(ZERO_FLOAT, hbm), max(ZERO_FLOAT, mte2)
    return replace(
        op,
        hbm_bytes=hbm,
        mte2_bytes=mte2,
        metadata=tuple(sorted(metadata.items(), key=lambda pair: pair[FIRST])),
        notes=tuple(getattr(op, "notes", ()) or ()) + (
            "prefix_read_plan_applied" if plan.read_reuse_applied
            else "prefix_read_plan_not_applied",),
    )
