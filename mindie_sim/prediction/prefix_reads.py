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
    q = int(access.q_len)
    if q <= 0:
        return ZERO_FLOAT
    if int(access.logical_rows) > 0:
        return float(int(access.logical_rows))
    context = max(0, int(access.context_tokens))
    win = (
        max(0, int(access.window_tokens))
        if access.window_tokens is not None and int(access.window_tokens) > 0
        else 0
    )
    if access.selected_tokens:
        selected = _merged_intervals(access.selected_tokens)
        total = 0
        for index in range(q):
            hi = context + index + 1
            lo = max(0, hi - win) if win > 0 else 0
            total += _merged_interval_rows(_intersect_intervals(selected, lo, hi))
        return float(total)
    total = 0
    for index in range(q):
        horizon = context + index + 1
        if win > 0:
            horizon = min(horizon, win)
        total += horizon
    return float(total)


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


def _intersect_interval_sets(
    left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    a, b = _merged_intervals(left), _merged_intervals(right)
    i = j = 0
    out: list[tuple[int, int]] = []
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0])
        hi = min(a[i][1], b[j][1])
        if hi > lo:
            out.append((lo, hi))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tuple(out)


def _subtract_interval_sets(
    left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    a, b = _merged_intervals(left), _merged_intervals(right)
    if not a or not b:
        return a
    out: list[tuple[int, int]] = []
    j = 0
    for lo, hi in a:
        cur = lo
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < hi:
            if b[k][0] > cur:
                out.append((cur, b[k][0]))
            cur = max(cur, b[k][1])
            if cur >= hi:
                break
            k += 1
        if cur < hi:
            out.append((cur, hi))
    return tuple(out)


def _take_interval_rows(
    intervals: Sequence[tuple[int, int]], limit: int,
) -> tuple[tuple[int, int], ...]:
    remaining = max(0, int(limit))
    if remaining <= 0:
        return ()
    out: list[tuple[int, int]] = []
    for lo, hi in _merged_intervals(intervals):
        if remaining <= 0:
            break
        take = min(hi - lo, remaining)
        if take > 0:
            out.append((lo, lo + take))
            remaining -= take
    return tuple(out)


def _parse_materialized_entry(entry: Any, upper: int) -> tuple[tuple[int, int], ...]:
    if entry is None or upper <= 0:
        return ()
    if isinstance(entry, Mapping):
        if "materialized_intervals" in entry or "intervals" in entry:
            raw = entry.get("materialized_intervals", entry.get("intervals"))
            return _parse_materialized_entry(raw, upper)
        limit = entry.get("materialized_tokens", entry.get("valid_tokens"))
        if limit is None:
            return ((0, upper),)
        return _parse_materialized_entry(limit, upper)
    if isinstance(entry, (int, float)) and not isinstance(entry, bool):
        bound = max(0, min(upper, int(entry)))
        return ((0, bound),) if bound > 0 else ()
    if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
        spans = tuple(entry)
        if not spans:
            return ()
        if all(isinstance(item, Sequence) and not isinstance(item, (str, bytes))
               and len(item) == PAIR for item in spans):
            return _intersect_intervals(spans, 0, upper)
    return ((0, upper),)


def _materialized_prefix_intervals(
    access: CacheAccess, cache_state: Mapping[str, Any] | None,
) -> tuple[tuple[int, int], ...]:
    declared = max(0, int(access.prefix_tokens))
    if declared <= 0:
        return ()
    if cache_state is None:
        return ((0, declared),)
    entry = cache_state.get(access.prefix_key, cache_state.get("*"))
    return _parse_materialized_entry(entry, declared)


def _resolve_shared_intervals(
    access: CacheAccess, cache_state: Mapping[str, Any] | None,
) -> tuple[tuple[int, int], ...]:
    if not access.shared_intervals:
        return _materialized_prefix_intervals(access, cache_state)
    raw_shared = _merged_intervals(access.shared_intervals)
    if cache_state is None:
        return raw_shared
    upper = max(
        (hi for _lo, hi in raw_shared),
        default=max(0, int(access.prefix_tokens)),
    )
    entry = cache_state.get(access.prefix_key, cache_state.get("*"))
    return _intersect_interval_sets(
        raw_shared, _parse_materialized_entry(entry, upper),
    )


def _resolve_access_partition(
    access: CacheAccess, rows: float, cache_state: Mapping[str, Any] | None,
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    """Partition one query's unique row span into shared intervals and private rows."""
    q_len = int(access.q_len)
    if q_len <= 0 or rows <= ZERO_FLOAT:
        return 0, 0, ()
    context = max(0, int(access.context_tokens))
    end = context + q_len
    if access.window_tokens is not None:
        win = max(0, int(access.window_tokens))
        windowed = win > 0
        start = max(0, context + 1 - win) if windowed else 0
    else:
        declared_prefix = min(max(0, int(access.prefix_tokens)), context)
        windowed = (
            not access.selected_tokens
            and not access.shared_intervals
            and rows < end
            and declared_prefix < int(rows)
        )
        start = max(0, end - int(rows)) if windowed else 0
    visible_span = max(0, end - start)
    if int(access.unique_rows) > 0:
        span = int(access.unique_rows)
    elif int(access.logical_rows) > 0:
        span = min(int(rows), visible_span)
    else:
        span = visible_span
    raw_shared = _resolve_shared_intervals(access, cache_state)
    if access.selected_tokens:
        visible_sel = _intersect_intervals(access.selected_tokens, start, end)
        shared_iv = _intersect_interval_sets(visible_sel, raw_shared)
        priv_iv = _subtract_interval_sets(visible_sel, shared_iv)
        return (
            _merged_interval_rows(shared_iv),
            _merged_interval_rows(priv_iv),
            shared_iv,
        )
    if access.shared_intervals:
        scoped = _intersect_intervals(raw_shared, start, end) if windowed else raw_shared
        intervals = _take_interval_rows(scoped, span)
        shared = _merged_interval_rows(intervals)
        return shared, max(0, span - shared), intervals
    if windowed:
        shared_iv = _intersect_intervals(raw_shared, start, end)
        intervals = _take_interval_rows(shared_iv, span)
        shared = _merged_interval_rows(intervals)
        return shared, max(0, span - shared), intervals
    limit = int(access.logical_rows) if int(access.logical_rows) > 0 else context
    shared_iv = _intersect_intervals(raw_shared, 0, max(0, limit))
    intervals = _take_interval_rows(shared_iv, span)
    shared = _merged_interval_rows(intervals)
    return shared, max(0, span - shared), intervals


def _l2_capacity_bytes(geometry: Any) -> float | None:
    if geometry is None:
        return None
    if isinstance(geometry, Mapping):
        raw = geometry.get("effective_l2_bytes", geometry.get("l2_capacity_bytes"))
    else:
        raw = getattr(
            geometry, "effective_l2_bytes", getattr(geometry, "l2_capacity_bytes", None)
        )
    if (isinstance(raw, (int, float)) and not isinstance(raw, bool)
            and math.isfinite(float(raw)) and float(raw) >= ZERO_FLOAT):
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
        + float(access.scale_bytes) * float(max(0, int(access.q_len)))
        for access in accesses)
    query_reads = math.fsum(
        float(max(0, int(access.q_len))) * float(access.row_bytes)
        for access in accesses)
    scale_total = math.fsum(
        float(access.scale_bytes) * float(max(0, int(access.q_len)))
        for access in accesses)
    if not accesses or mode == MODE_PER_SEQUENCE:
        return PrefixReadPlan(
            logical, logical, logical, query_reads, scale_total, 0, 0,
            REUSE_PER_SEQUENCE, False, NO_REUSE_REASONS["mode_not_shared"])
    shared_candidates = [
        access for access in accesses
        if int(access.prefix_tokens) > 0 and int(access.q_len) > 0
    ]
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
    unique_payload = ZERO_FLOAT
    unique_scale = ZERO_FLOAT
    core = ZERO_FLOAT
    groups: dict[tuple[int, str, str, float], list[tuple[int, int]]] = {}
    scale_groups: dict[
        tuple[int, str, str, float], tuple[list[tuple[int, int]], float, float]
    ] = {}
    for access in accesses:
        q = max(0, int(access.q_len))
        rows = _logical_rows(access)
        acc_scale = float(access.scale_bytes) * float(q)
        shared, private, intervals = _resolve_access_partition(
            access, rows, cache_state,
        )
        unique_payload += float(private) * float(access.row_bytes)
        core += rows * float(access.row_bytes) + acc_scale
        scale_per_row = (acc_scale / rows) if rows > ZERO_FLOAT else ZERO_FLOAT
        unique_scale += float(private) * scale_per_row
        if shared > 0 and intervals:
            key = (int(access.layer), str(access.component),
                   str(access.prefix_key), float(access.row_bytes))
            groups.setdefault(key, []).extend(intervals)
            if scale_per_row > ZERO_FLOAT:
                entry = scale_groups.get(key)
                if entry is None:
                    scale_groups[key] = (
                        list(intervals), float(shared), float(shared) * scale_per_row,
                    )
                else:
                    entry_iv, entry_rows, entry_scale = entry
                    entry_iv.extend(intervals)
                    scale_groups[key] = (
                        entry_iv,
                        entry_rows + float(shared),
                        entry_scale + float(shared) * scale_per_row,
                    )
    for key, intervals in groups.items():
        shared_rows = _merged_interval_rows(intervals)
        unique_payload += float(shared_rows) * float(key[3])
    for _key, (s_intervals, s_rows, s_scale) in scale_groups.items():
        if s_rows > ZERO_FLOAT:
            merged_s = _merged_interval_rows(s_intervals)
            unique_scale += s_scale * (float(merged_s) / s_rows)
    working_set = unique_payload + unique_scale
    l2_cap = _l2_capacity_bytes(geometry)
    if kernel_reuse == REUSE_PAGED_L2 and l2_cap is not None:
        if working_set > ZERO_FLOAT:
            retained = min(working_set, l2_cap)
            spill_ratio = max(ZERO_FLOAT, ONE_FLOAT - retained / working_set)
            unique = working_set + max(ZERO_FLOAT, logical - working_set) * spill_ratio
        elif l2_cap == ZERO_FLOAT:
            unique = logical
        else:
            unique = working_set
    else:
        unique = working_set
    unique = min(logical, unique)
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
