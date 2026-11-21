"""Exact periodic and strided byte-run coalescing, residency splitting and overlap.

[Role] Coalesce adjacent equal-stride address runs and evaluate exact residency
  and overlap queries without expanding large coprime common periods.
[Invariants] Output byte sets are exact and pairwise disjoint across warm/cold
  classifications; sparse residency intersections stay O(sparse rows) in memory.
[Timeouts & Locks] Pure bounded in-memory arithmetic; no I/O, threads or locks.
[Side-Effect Policy] Pure value transformations with zero global mutation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from heapq import merge
from itertools import groupby
import math
from math import gcd, lcm
from typing import Any

from .memory_physics import (
    AddressRun,
    PeriodicSectorRun,
    SectorCoverage,
    _MAX_EXPLICIT_SECTOR_INTERVALS,
    _ceil_div,
    _intersect_byte_intervals,
    _measure_byte_intervals,
    _merge_byte_intervals,
)


def _range_run(lo: int, hi: int) -> AddressRun:
    return AddressRun(lo, 1, hi - lo, hi - lo)


def _run_end(run: AddressRun) -> int:
    return run.base_byte + (run.rows - 1) * run.stride_bytes + run.row_bytes


def _coverage_runs(coverage: SectorCoverage):
    yield from (_range_run(lo, hi) for lo, hi in coverage.intervals)
    for run in coverage.periodic:
        # A sector-aligned phase repeats after the row's byte offset realigns.
        phase_rows = run.sector_bytes // gcd(run.sector_bytes, run.stride_bytes)
        stride = phase_rows * run.stride_bytes
        for phase in range(min(phase_rows, run.rows)):
            lo, hi = run._row_bounds(phase)
            rows = (run.rows - 1 - phase) // phase_rows + 1
            yield AddressRun(lo, rows, hi - lo, stride)


def _coalesce_runs(runs) -> tuple[AddressRun, ...]:
    """Merge overlapping single spans and adjacent equal-stride periodic runs."""
    compacted: list[AddressRun] = []
    multi_tails: dict[tuple[int, int, int], int] = {}
    for run in sorted(runs, key=lambda item: (item.base_byte, item.stride_bytes, item.row_bytes)):
        if not run.rows or not run.row_bytes:
            continue
        if run.rows == 1 or run.stride_bytes <= run.row_bytes:
            run = _range_run(run.base_byte, _run_end(run))
        if run.rows > 1:
            chain_key = (run.row_bytes, run.stride_bytes, run.base_byte)
            prev_idx = multi_tails.pop(chain_key, None)
            if prev_idx is not None:
                prev = compacted[prev_idx]
                merged = replace(prev, rows=prev.rows + run.rows)
                compacted[prev_idx] = merged
                multi_tails[(merged.row_bytes, merged.stride_bytes,
                             merged.base_byte + merged.rows * merged.stride_bytes)] = prev_idx
                continue
        if compacted:
            previous = compacted[-1]
            if previous.rows == 1 and run.rows == 1 and run.base_byte <= _run_end(previous):
                compacted[-1] = _range_run(previous.base_byte, max(_run_end(previous), _run_end(run)))
                continue
            stride = previous.stride_bytes if previous.rows > 1 else run.stride_bytes
            if (max(previous.rows, run.rows) > 1 and previous.row_bytes == run.row_bytes
                    and (run.rows == 1 or run.stride_bytes == stride)
                    and previous.base_byte + previous.rows * stride == run.base_byte):
                if previous.rows > 1:
                    multi_tails.pop((previous.row_bytes, previous.stride_bytes,
                                     previous.base_byte + previous.rows * previous.stride_bytes), None)
                merged = replace(previous, rows=previous.rows + run.rows, stride_bytes=stride)
                compacted[-1] = merged
                multi_tails[(merged.row_bytes, merged.stride_bytes,
                             merged.base_byte + merged.rows * merged.stride_bytes)] = len(compacted) - 1
                continue
        idx = len(compacted)
        compacted.append(run)
        if run.rows > 1:
            multi_tails[(run.row_bytes, run.stride_bytes,
                         run.base_byte + run.rows * run.stride_bytes)] = idx
    return tuple(compacted)


def _sweep(events):
    """Yield consecutive byte spans with the labels active throughout each span."""
    active, previous = Counter(), 0
    for offset, changes in groupby(events, key=lambda event: event[0]):
        if active and offset > previous:
            yield previous, offset, active
        for _, label, delta in changes:
            active[label] += delta
            if not active[label]:
                del active[label]
        previous = offset


def _window_rows(run, lo, hi):
    return range(max(0, (lo - run.base_byte - run.row_bytes) // run.stride_bytes + 1),
                 min(run.rows, _ceil_div(hi - run.base_byte, run.stride_bytes)))


def _row_events(run, label, lo, hi):
    for row in _window_rows(run, lo, hi):
        start = run.base_byte + row * run.stride_bytes
        yield max(lo, start), label, 1
        yield min(hi, start + run.row_bytes), label, -1


def _classified_spans(events, classify):
    previous = None
    for lo, hi, active in _sweep(events):
        state = classify(active)
        if previous is not None and previous[1] == lo and previous[2] == state:
            previous = previous[0], hi, state
            continue
        if previous is not None:
            yield previous
        previous = (lo, hi, state) if state is not None else None
    if previous is not None:
        yield previous


def _partition_runs(tagged_runs, classify):
    """Apply a byte-set operation across repeating spans without expanding their rows."""
    events = []
    for run, label in dict.fromkeys(tagged_runs):
        end = _run_end(run)
        if run.rows == 1 or run.stride_bytes <= run.row_bytes:
            run = _range_run(run.base_byte, end)
        events.extend(((run.base_byte, (run, label), 1), (end, (run, label), -1)))
    for lo, hi, active in _sweep(sorted(events, key=lambda event: event[0])):
        if all(run.rows == 1 for run, _ in active):
            if (state := classify({label for _, label in active})) is not None:
                yield state, _range_run(lo, hi)
            continue
        # Each active run repeats over its bounding span. Only one common
        # period needs scanning, with the final repeat clipped to the span.
        period = lcm(*(run.stride_bytes for run, _ in active if run.rows > 1))
        end = min(hi, lo + period)
        streams = (_row_events(run, label, lo, end) for run, label in active)
        for begin, finish, state in _classified_spans(merge(*streams, key=lambda event: event[0]), classify):
            rows = (hi - finish) // period + 1
            width = finish - begin
            yield state, AddressRun(begin, rows, width, period) if rows > 1 else _range_run(begin, finish)
            if (tail := hi - (begin + rows * period)) > 0:
                yield state, _range_run(hi - tail, hi)


def _sparse_window(run, masks) -> bool:
    if run.rows <= 1 or run.stride_bytes <= run.row_bytes:
        return False
    period = lcm(run.stride_bytes, *(r.stride_bytes for r in masks if r.rows > 1))
    rows = sum(len(_window_rows(mask, run.base_byte, _run_end(run))) for mask in masks)
    return min(run.rows, period // run.stride_bytes) > max(_MAX_EXPLICIT_SECTOR_INTERVALS, rows)


def _disjoint_periodic(runs) -> bool:
    if len(runs) <= 1:
        return True
    ordered = sorted(runs, key=lambda r: r._bounds())
    return all(left._bounds()[1] <= right._bounds()[0] for left, right in zip(ordered, ordered[1:]))


def _coalesce_periodic_runs(runs: tuple) -> tuple:
    """Merge consecutive identical-geometry periodic runs into one descriptor."""
    if len(runs) <= 1:
        return runs
    ordered = sorted(runs, key=lambda r: (r.stride_bytes, r.row_bytes, r.sector_bytes, r.base_byte))
    merged = [ordered[0]]
    for run in ordered[1:]:
        prev = merged[-1]
        if (prev.stride_bytes == run.stride_bytes > 0
                and prev.row_bytes == run.row_bytes
                and prev.sector_bytes == run.sector_bytes
                and prev.base_byte + prev.rows * prev.stride_bytes == run.base_byte):
            merged[-1] = replace(prev, rows=prev.rows + run.rows)
        else:
            merged.append(run)
    return tuple(merged)


def _two_color_periodic(runs) -> tuple[tuple, tuple] | None:
    """Partition periodic runs into at most two mutually bounding-disjoint groups."""
    ordered = sorted(runs, key=lambda r: r._bounds())
    group_a: list = []
    group_b: list = []
    end_a = -1
    end_b = -1
    for run in ordered:
        lo, hi = run._bounds()
        if hi <= lo:
            continue
        if lo >= end_a:
            group_a.append(run)
            end_a = hi
        elif lo >= end_b:
            group_b.append(run)
            end_b = hi
        else:
            return None
    return tuple(group_a), tuple(group_b)


def _periodic_subrun(sub, sup) -> bool:
    """True when every row of ``sub`` is an exact row of ``sup``."""
    if sub == sup:
        return True
    if (sub.stride_bytes != sup.stride_bytes or sub.stride_bytes <= 0
            or sub.row_bytes != sup.row_bytes or sub.sector_bytes != sup.sector_bytes):
        return False
    delta = sub.base_byte - sup.base_byte
    return delta >= 0 and delta % sup.stride_bytes == 0 and (delta // sup.stride_bytes) + sub.rows <= sup.rows


def _prune_periodic_subruns(runs: tuple) -> tuple:
    """Drop periodic descriptors completely covered by another run in ``runs``."""
    unique = tuple(dict.fromkeys(runs))
    if len(unique) <= 1:
        return unique
    kept = []
    for idx, run in enumerate(unique):
        if any(j != idx and _periodic_subrun(run, other) for j, other in enumerate(unique)):
            continue
        kept.append(run)
    return tuple(kept)


def _coverage_overlap(left: SectorCoverage, right: SectorCoverage) -> int:
    if left.empty or right.empty:
        return 0
    left_c = _prune_periodic_subruns(_coalesce_periodic_runs(left.periodic)) if left.periodic else ()
    right_c = _prune_periodic_subruns(_coalesce_periodic_runs(right.periodic)) if right.periodic else ()
    if not left.intervals and left_c and all(any(_periodic_subrun(l, r) for r in right_c) for l in left_c):
        return _merged_bytes(SectorCoverage((), left_c))
    if not right.intervals and right_c and all(any(_periodic_subrun(r, l) for l in left_c) for r in right_c):
        return _merged_bytes(SectorCoverage((), right_c))
    if not left.intervals and not right.intervals and left_c and right_c:
        left_p = tuple(
            l for l in left_c
            if any(l._bounds()[0] < r._bounds()[1] and r._bounds()[0] < l._bounds()[1] for r in right_c)
        )
        if not left_p:
            return 0
        right_p = tuple(
            r for r in right_c
            if any(l._bounds()[0] < r._bounds()[1] and r._bounds()[0] < l._bounds()[1] for l in left_p)
        )
        if not right_p:
            return 0
        if len(left_p) == 1 and len(right_p) == 1:
            if left_p[0] == right_p[0]:
                return left_p[0].total_bytes()
            dense, sparse = sorted((left_p[0], right_p[0]), key=lambda r: r.stride_bytes)
            d_lo, d_hi = dense._bounds()
            s_lo, s_hi = sparse._bounds()
            if d_hi <= s_lo or s_hi <= d_lo:
                return 0
            if sparse.stride_bytes >= sparse.row_bytes + sparse.sector_bytes and _sparse_window(dense, (sparse,)):
                rows = _window_rows(sparse, d_lo - sparse.sector_bytes, d_hi + sparse.sector_bytes)
                return sum(dense.overlap_with_interval(*sparse._row_bounds(row)) for row in rows)
        elif _disjoint_periodic(left_p) and len(left_p) > 1:
            return sum(
                _coverage_overlap(SectorCoverage((), (l,)), SectorCoverage((), right_p))
                for l in left_p
            )
        elif _disjoint_periodic(right_p) and len(right_p) > 1:
            return sum(
                _coverage_overlap(SectorCoverage((), left_p), SectorCoverage((), (r,)))
                for r in right_p
            )
    return left.overlap_bytes(right)


def _merged_bytes(coverage: SectorCoverage) -> int:
    if coverage.periodic:
        runs = _prune_periodic_subruns(_coalesce_periodic_runs(coverage.periodic))
        if not coverage.intervals:
            if len(runs) == 1:
                return runs[0].total_bytes()
            colored = _two_color_periodic(runs)
            if colored is not None:
                group_a, group_b = colored
                total = sum(r.total_bytes() for r in group_a) + sum(r.total_bytes() for r in group_b)
                if group_a and group_b:
                    total -= sum(
                        _coverage_overlap(SectorCoverage((), (a,)), SectorCoverage((), (b,)))
                        for a in group_a for b in group_b
                    )
                return total
        elif _disjoint_periodic(runs):
            ivs = coverage.intervals
            return sum(hi - lo for lo, hi in ivs) + sum(
                max(0, r.total_bytes() - sum(r.overlap_with_interval(lo, hi) for lo, hi in ivs))
                for r in runs
            )
    return coverage.measure_bytes()


def _clip_rows(run: AddressRun, lo: int, hi: int):
    rows = _window_rows(run, lo, hi)
    if rows and hi > lo:
        first = run.base_byte + rows.start * run.stride_bytes
        yield _range_run(max(lo, first), min(hi, first + run.row_bytes))
        if len(rows) > 2:
            yield replace(run, base_byte=first + run.stride_bytes, rows=len(rows) - 2)
        if len(rows) > 1:
            last = run.base_byte + (rows.stop - 1) * run.stride_bytes
            yield _range_run(last, min(hi, last + run.row_bytes))


def _disjoint_components(runs: tuple[AddressRun, ...]):
    ordered = sorted(runs, key=lambda item: (item.base_byte, _run_end(item)))
    components: list[tuple[AddressRun, ...]] = []
    group = [ordered[0]]
    limit = _run_end(ordered[0])
    for run in ordered[1:]:
        if run.base_byte >= limit:
            components.append(tuple(group))
            group = [run]
            limit = _run_end(run)
        else:
            group.append(run)
            limit = max(limit, _run_end(run))
    components.append(tuple(group))
    return tuple(components)


def _sparse_residency_run(run: AddressRun, masks: tuple[AddressRun, ...]):
    lo, hi = run.base_byte, _run_end(run)
    streams = (_row_events(mask, True, lo, hi) for mask in masks)
    for begin, end, _ in _classified_spans(merge(*streams, key=lambda e: e[0]), bool):
        warm = tuple(_clip_rows(run, begin, end))
        if not warm:
            continue
        yield from ((False, part) for part in _clip_rows(run, lo, begin))
        yield from ((True, part) for part in warm)
        lo = end
    yield from ((False, part) for part in _clip_rows(run, lo, hi))


def _residency_runs(data, masks):
    data_runs = _coalesce_runs(data)
    mask_runs = _coalesce_runs(masks)
    if not data_runs:
        return
    if not mask_runs:
        yield from ((False, run) for run in data_runs)
        return
    for component in _disjoint_components(data_runs):
        if len(component) == 1:
            run = component[0]
            if _sparse_window(run, mask_runs):
                yield from _sparse_residency_run(run, mask_runs)
            else:
                yield from _partition_runs(
                    ((run, False), *((m, True) for m in mask_runs)),
                    lambda labels: True in labels if False in labels else None,
                )
            continue
        if any(_sparse_window(run, mask_runs) for run in component):
            clean = _coalesce_runs(
                run for _, run in _partition_runs(((r, True) for r in component), lambda _: True)
            )
            for run in clean:
                if _sparse_window(run, mask_runs):
                    yield from _sparse_residency_run(run, mask_runs)
                else:
                    yield from _partition_runs(
                        ((run, False), *((m, True) for m in mask_runs)),
                        lambda labels: True in labels if False in labels else None,
                    )
        else:
            yield from _partition_runs(
                (*((r, False) for r in component), *((m, True) for m in mask_runs)),
                lambda labels: True in labels if False in labels else None,
            )


def _measure_periodic_combo(
    left: tuple[PeriodicSectorRun, ...], right: tuple[PeriodicSectorRun, ...], *,
    intersect: bool, extra_left_ivs: tuple[tuple[int, int], ...] = (),
    extra_right_ivs: tuple[tuple[int, int], ...] = (),
) -> int:
    """Exact byte measure of union (``intersect=False``) or intersection of runs + intervals."""
    left_c = tuple(dict.fromkeys(r.canonical() for r in left if r.rows > 0 and r.row_bytes > 0))
    right_c = tuple(dict.fromkeys(r.canonical() for r in right if r.rows > 0 and r.row_bytes > 0))
    if not intersect and not left_c and not right_c:
        return _measure_byte_intervals(_merge_byte_intervals((*extra_left_ivs, *extra_right_ivs)))
    if intersect and ((not left_c and not extra_left_ivs) or (not right_c and not extra_right_ivs)):
        return 0
    if not intersect and len(left_c) == 1 and not right_c:
        r0, ivs = left_c[0], _merge_byte_intervals((*extra_left_ivs, *extra_right_ivs))
        return _measure_byte_intervals(ivs) + max(0, r0.total_bytes() - sum(r0.overlap_with_interval(a, b) for a, b in ivs))
    if intersect and len(left_c) == 1 and len(right_c) == 1 and not extra_left_ivs and not extra_right_ivs:
        a, b = left_c[0], right_c[0]
        if a == b:
            return a.total_bytes()
        if (a.sector_bytes == b.sector_bytes and a.stride_bytes == b.stride_bytes > 0
                and a.stride_bytes % a.sector_bytes == 0
                and (a.rows == 1 or a.row_bytes < a.stride_bytes)
                and (b.rows == 1 or b.row_bytes < b.stride_bytes)):
            st, w1, w2 = a.stride_bytes, a.row_bytes, b.row_bytes
            q, rem = divmod(b.base_byte - a.base_byte, st)
            total = 0
            for s in range(math.floor((-rem - w2) / st), math.ceil((w1 - rem) / st) + 1):
                ov_w = min(w1, rem + s * st + w2) - max(0, rem + s * st)
                k_lo, k_hi = max(0, q - s), min(a.rows - 1, b.rows - 1 + q - s)
                if ov_w > 0 and k_hi >= k_lo:
                    total += (k_hi - k_lo + 1) * ov_w
            return total
    events: list[tuple[int, int, str, Any]] = []
    for kind, src in (("li", extra_left_ivs), ("ri", extra_right_ivs)):
        for lo, hi in src:
            if hi > lo:
                events.extend(((lo, 1, kind, None), (hi, -1, kind, None)))
    for kind, src in (("lr", left_c), ("rr", right_c)):
        for r in src:
            b_lo, b_hi = r._bounds()
            if b_hi > b_lo:
                events.extend(((b_lo, 1, kind, r), (b_hi, -1, kind, r)))
    if not events:
        return 0
    events.sort(key=lambda item: item[0])

    def _periodic_seg(act, s_lo, s_hi, eval_fn):
        period = math.lcm(*(r._period() for r in act))
        in_lo = min(s_hi, max(s_lo, max(r._bounds()[0] + period for r in act)))
        in_hi = max(in_lo, min(s_hi, min(r._bounds()[1] - period for r in act)))
        acc = eval_fn(s_lo, in_lo) if in_lo > s_lo else 0
        if in_hi > in_lo:
            full, rem = divmod(in_hi - in_lo, period)
            acc += (full * eval_fn(in_lo, in_lo + period) if full else 0) + (eval_fn(in_lo + full * period, in_hi) if rem else 0)
        return acc + (eval_fn(in_hi, s_hi) if s_hi > in_hi else 0)

    l_iv = r_iv = total = idx = 0
    l_set: set[PeriodicSectorRun] = set()
    r_set: set[PeriodicSectorRun] = set()
    n_ev, prev_x = len(events), events[0][0]
    while idx < n_ev:
        cur_x = events[idx][0]
        if cur_x > prev_x:
            s_lo, s_hi = prev_x, cur_x
            l_cov, r_cov = l_iv > 0, r_iv > 0
            l_act, r_act = (() if l_cov else tuple(l_set)), (() if r_cov else tuple(r_set))
            if not intersect:
                if l_cov or r_cov:
                    total += s_hi - s_lo
                elif act := tuple(dict.fromkeys((*l_act, *r_act))):
                    total += _periodic_seg(act, s_lo, s_hi, lambda w0, w1, _a=act: _measure_byte_intervals(
                        _merge_byte_intervals([iv for r in _a for iv in r._window_intervals(w0, w1)])
                    ))
            elif (l_cov or l_act) and (r_cov or r_act):
                if l_cov and r_cov:
                    total += s_hi - s_lo
                else:
                    act = tuple(dict.fromkeys((*l_act, *r_act)))
                    total += _periodic_seg(act, s_lo, s_hi, lambda w0, w1, _lc=l_cov, _rc=r_cov, _la=l_act, _ra=r_act: _measure_byte_intervals(_intersect_byte_intervals(
                        ((w0, w1),) if _lc else _merge_byte_intervals([iv for r in _la for iv in r._window_intervals(w0, w1)]),
                        ((w0, w1),) if _rc else _merge_byte_intervals([iv for r in _ra for iv in r._window_intervals(w0, w1)]),
                    )))
        while idx < n_ev and events[idx][0] == cur_x:
            _, delta, kind, obj = events[idx]
            if kind == "li":
                l_iv += delta
            elif kind == "ri":
                r_iv += delta
            elif kind == "lr":
                l_set.add(obj) if delta > 0 else l_set.discard(obj)
            else:
                r_set.add(obj) if delta > 0 else r_set.discard(obj)
            idx += 1
        prev_x = cur_x
    return total


__all__ = [
    "_classified_spans",
    "_clip_rows",
    "_coalesce_runs",
    "_coverage_overlap",
    "_coverage_runs",
    "_measure_periodic_combo",
    "_merged_bytes",
    "_partition_runs",
    "_range_run",
    "_residency_runs",
    "_row_events",
    "_run_end",
    "_sparse_window",
    "_sweep",
    "_window_rows",
]
