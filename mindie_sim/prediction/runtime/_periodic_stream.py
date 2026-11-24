"""Bounded-memory streaming and modular-jump exact measures for periodic runs.

[Role] Evaluate exact byte union and intersection measures over multiple
  ``PeriodicSectorRun`` descriptors inside a window ``[lo, hi)`` without
  materializing O(rows) interval lists in memory.
[Invariants] Peak auxiliary memory is O(K) in the number of active runs K
  (O(1) for K <= 3), and output byte counts are bit-for-bit identical to
  explicit sector-interval merging.
[Timeouts & Locks] Pure bounded in-memory arithmetic; no I/O, threads or locks.
[Side-Effect Policy] Pure value functions with zero global mutation.
"""
from __future__ import annotations

from bisect import bisect_left
from collections.abc import Sequence
import math

from .memory_physics import PeriodicSectorRun

_PAIR_RUNS = 2
_TRIPLE_RUNS = 3
_HALF_MOD_DIVISOR = 2
_BISECT_OVERLAP_RATIO = 16


def _aligned_intervals_bytes(
    intervals: tuple[tuple[int, int], ...], line_bytes: int,
) -> int:
    """Exact byte span of sorted disjoint intervals rounded outward to ``line_bytes``."""
    if not intervals:
        return 0
    total = 0
    cur_lo = -1
    cur_hi = -1
    step = line_bytes - 1
    for lo, hi in intervals:
        al = (lo // line_bytes) * line_bytes
        ah = ((hi + step) // line_bytes) * line_bytes
        if al <= cur_hi:
            if ah > cur_hi:
                cur_hi = ah
        else:
            if cur_hi > cur_lo:
                total += cur_hi - cur_lo
            cur_lo, cur_hi = al, ah
    return total + (cur_hi - cur_lo if cur_hi > cur_lo else 0)



def _merge_interval_slices(
    left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Linear two-pointer merge of two sorted disjoint interval sequences."""
    nl, nr = len(left), len(right)
    merged: list[tuple[int, int]] = []
    i = j = 0
    while i < nl or j < nr:
        if j >= nr or (i < nl and left[i][0] <= right[j][0]):
            lo, hi = left[i]
            i += 1
        else:
            lo, hi = right[j]
            j += 1
        if merged and lo <= merged[-1][1]:
            if hi > merged[-1][1]:
                merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    return merged


def _plan_canonical_list_merge(
    left: list[tuple[int, int]],
    right: tuple[tuple[int, int], ...],
    left_fp: int | None,
    line_bytes: int | None,
) -> tuple[int, int, list[tuple[int, int]], int | None]:
    """Plan an in-place slice merge into canonical ``left`` and incremental ``line_bytes`` footprint."""
    if not right:
        fp = (None if line_bytes is None
              else (_aligned_intervals_bytes(left, line_bytes) if left_fp is None else left_fp))
        return 0, 0, [], fp
    nl = len(left)
    if not nl:
        fp = None if line_bytes is None else _aligned_intervals_bytes(right, line_bytes)
        return 0, 0, list(right), fp
    r_min, r_max = right[0][0], right[-1][1]
    if left[-1][1] < r_min:
        lo_idx = hi_idx = nl
        merged_mid = list(right)
    elif r_max < left[0][0]:
        lo_idx = hi_idx = 0
        merged_mid = list(right)
    else:
        lo_idx = bisect_left(left, (r_min, 0))
        if lo_idx > 0 and left[lo_idx - 1][1] >= r_min:
            lo_idx -= 1
        hi_idx = bisect_left(left, (r_max + 1, 0), lo_idx)
        sub_left = left[lo_idx:hi_idx]
        merged_mid = list(right) if not sub_left else _merge_interval_slices(sub_left, right)
    if line_bytes is None:
        return lo_idx, hi_idx, merged_mid, None
    if left_fp is None:
        full = left[:lo_idx] + merged_mid + left[hi_idx:]
        return lo_idx, hi_idx, merged_mid, _aligned_intervals_bytes(full, line_bytes)
    lo_fp = bisect_left(left, (max(0, r_min - line_bytes), 0))
    if lo_fp > 0 and left[lo_fp - 1][1] + line_bytes >= r_min:
        lo_fp -= 1
    hi_fp = bisect_left(left, (r_max + line_bytes + 1, 0), hi_idx)
    old_sub_fp = _aligned_intervals_bytes(left[lo_fp:hi_fp], line_bytes)
    new_win = left[lo_fp:lo_idx] + merged_mid + left[hi_idx:hi_fp]
    new_sub_fp = _aligned_intervals_bytes(new_win, line_bytes)
    return lo_idx, hi_idx, merged_mid, left_fp - old_sub_fp + new_sub_fp


def _merge_canonical_pairs(
    left: tuple[tuple[int, int], ...], right: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Merge two canonical sorted disjoint interval tuples using bisect windowing."""
    if not left:
        return right
    if not right:
        return left
    if left[-1][1] < right[0][0]:
        return left + right
    if right[-1][1] < left[0][0]:
        return right + left
    if len(left) < len(right):
        left, right = right, left
    nl, nr = len(left), len(right)
    if nl > nr * _HALF_MOD_DIVISOR:
        r_min, r_max = right[0][0], right[-1][1]
        lo_idx = bisect_left(left, (r_min, 0))
        if lo_idx > 0 and left[lo_idx - 1][1] >= r_min:
            lo_idx -= 1
        hi_idx = bisect_left(left, (r_max + 1, 0), lo_idx)
        if lo_idx > 0 or hi_idx < nl:
            sub_left = left[lo_idx:hi_idx]
            if not sub_left:
                return left[:lo_idx] + right + left[hi_idx:]
            merged_mid = _merge_canonical_pairs(sub_left, right)
            return left[:lo_idx] + merged_mid + left[hi_idx:]
    return tuple(_merge_interval_slices(left, right))


def _overlap_canonical_intervals(
    left: tuple[tuple[int, int], ...], right: tuple[tuple[int, int], ...],
) -> int:
    """Exact overlap byte count between two canonical sorted disjoint interval tuples."""
    if not left or not right:
        return 0
    if left[-1][1] <= right[0][0] or right[-1][1] <= left[0][0]:
        return 0
    if len(left) > len(right):
        left, right = right, left
    nl, nr = len(left), len(right)
    l_min, l_max = left[0][0], left[-1][1]
    j0 = bisect_left(right, (l_min, 0))
    if j0 > 0 and right[j0 - 1][1] > l_min:
        j0 -= 1
    j1 = bisect_left(right, (l_max, 0), j0)
    if j0 >= j1:
        return 0
    total = 0
    if nl * _BISECT_OVERLAP_RATIO < (j1 - j0):
        idx = j0
        for lo, hi in left:
            idx = bisect_left(right, (lo, 0), idx, j1)
            if idx > j0 and right[idx - 1][1] > lo:
                idx -= 1
            while idx < j1 and right[idx][0] < hi:
                ov_hi = hi if hi < right[idx][1] else right[idx][1]
                ov_lo = lo if lo > right[idx][0] else right[idx][0]
                if ov_hi > ov_lo:
                    total += ov_hi - ov_lo
                idx += 1
        return total
    i, j = 0, j0
    while i < nl and j < j1:
        a_lo, a_hi = left[i]
        b_lo, b_hi = right[j]
        ov_lo = a_lo if a_lo > b_lo else b_lo
        ov_hi = a_hi if a_hi < b_hi else b_hi
        if ov_hi > ov_lo:
            total += ov_hi - ov_lo
        if a_hi < b_hi:
            i += 1
        else:
            j += 1
    return total


def _run_overlap_small_interval(run: PeriodicSectorRun, lo: int, hi: int) -> int:
    """Exact byte overlap of ``run`` with a small window ``[lo, hi)`` in O(1) memory."""
    if hi <= lo or run.rows <= 0 or run.row_bytes <= 0 or run.stride_bytes <= 0:
        return 0
    sec, stride, base, rb = run.sector_bytes, run.stride_bytes, run.base_byte, run.row_bytes
    if hi - 1 < base:
        return 0
    num_lo = lo - sec - base - rb
    k_lo = 0 if num_lo < 0 else (num_lo // stride) + 1
    k_hi = min(run.rows - 1, (hi - 1 - base) // stride)
    if k_lo > k_hi:
        return 0
    total = 0
    cur_lo = cur_hi = -1
    for step in range(k_lo, k_hi + 1):
        addr = base + step * stride
        s_k = (addr // sec) * sec
        e_k = ((addr + rb + sec - 1) // sec) * sec
        c_lo = lo if s_k < lo else s_k
        c_hi = hi if e_k > hi else e_k
        if c_hi <= c_lo:
            continue
        if cur_hi < 0:
            cur_lo, cur_hi = c_lo, c_hi
        elif c_lo <= cur_hi:
            if c_hi > cur_hi:
                cur_hi = c_hi
        else:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = c_lo, c_hi
    if cur_hi >= 0:
        total += cur_hi - cur_lo
    return total


def _iter_two_run_overlap(a: PeriodicSectorRun, b: PeriodicSectorRun, lo: int, hi: int):
    """Yield merged overlap intervals of ``a & b`` inside ``[lo, hi)`` in O(1) memory."""
    if hi <= lo or a.rows <= 0 or b.rows <= 0 or a.row_bytes <= 0 or b.row_bytes <= 0:
        return
    if b.stride_bytes > a.stride_bytes:
        a, b = b, a
    sa, sb = a.stride_bytes, b.stride_bytes
    ba, bb = a.base_byte, b.base_byte
    ra, rb = a.row_bytes, b.row_bytes
    seca, secb = a.sector_bytes, b.sector_bytes
    if sa <= 0 or sb <= 0 or hi - 1 < ba:
        return
    num_lo = lo - seca - ba - ra
    ka_lo = 0 if num_lo < 0 else (num_lo // sa) + 1
    ka_hi = min(a.rows - 1, (hi - 1 - ba) // sa)
    if ka_lo > ka_hi:
        return
    hi_mod = rb + seca + secb - 1
    lo_mod = ra + seca + secb - 1
    use_mod_jump = sb > hi_mod + lo_mod + 1
    step_mod = sa % sb
    half_sb = sb // _HALF_MOD_DIVISOR
    i = ka_lo
    cur_lo = cur_hi = -1
    while i <= ka_hi:
        if use_mod_jump:
            rem = (ba + i * sa - bb) % sb
            if hi_mod <= rem <= sb - lo_mod:
                if step_mod == 0:
                    break
                if step_mod <= half_sb:
                    dist = (sb - lo_mod + 1) - rem
                    i += (dist + step_mod - 1) // step_mod
                    continue
                dec = sb - step_mod
                dist = rem - (hi_mod - 1)
                i += (dist + dec - 1) // dec
                continue
        addr_a = ba + i * sa
        s_a = (addr_a // seca) * seca
        e_a = ((addr_a + ra + seca - 1) // seca) * seca
        c_lo = lo if s_a < lo else s_a
        c_hi = hi if e_a > hi else e_a
        if c_hi > c_lo and c_hi - 1 >= bb:
            n_lo_b = c_lo - secb - bb - rb
            jb_lo = 0 if n_lo_b < 0 else (n_lo_b // sb) + 1
            jb_hi = min(b.rows - 1, (c_hi - 1 - bb) // sb)
            for j in range(jb_lo, jb_hi + 1):
                addr_b = bb + j * sb
                s_b = (addr_b // secb) * secb
                e_b = ((addr_b + rb + secb - 1) // secb) * secb
                o_lo = c_lo if s_b < c_lo else s_b
                o_hi = c_hi if e_b > c_hi else e_b
                if o_hi > o_lo:
                    if cur_hi < 0:
                        cur_lo, cur_hi = o_lo, o_hi
                    elif o_lo <= cur_hi:
                        if o_hi > cur_hi:
                            cur_hi = o_hi
                    else:
                        yield cur_lo, cur_hi
                        cur_lo, cur_hi = o_lo, o_hi
        i += 1
    if cur_hi >= 0:
        yield cur_lo, cur_hi


def _measure_pair_overlap(a: PeriodicSectorRun, b: PeriodicSectorRun, lo: int, hi: int) -> int:
    """Exact byte overlap of two periodic runs in ``[lo, hi)`` with O(1) memory."""
    alo, ahi = a._bounds()
    blo, bhi = b._bounds()
    s_lo = max(lo, alo, blo)
    s_hi = min(hi, ahi, bhi)
    if s_hi <= s_lo:
        return 0
    period = math.lcm(a._period(), b._period())
    in_lo = min(s_hi, max(s_lo, max(alo + period, blo + period)))
    in_hi = max(in_lo, min(s_hi, min(ahi - period, bhi - period)))

    def _ev(w0: int, w1: int) -> int:
        return sum(y - x for x, y in _iter_two_run_overlap(a, b, w0, w1))

    acc = _ev(s_lo, in_lo) if in_lo > s_lo else 0
    if in_hi > in_lo:
        full, rem = divmod(in_hi - in_lo, period)
        acc += (full * _ev(in_lo, in_lo + period) if full else 0) + (_ev(in_lo + full * period, in_hi) if rem else 0)
    return acc + (_ev(in_hi, s_hi) if s_hi > in_hi else 0)


def _iter_union_stream(runs: tuple[PeriodicSectorRun, ...], lo: int, hi: int):
    """Stream merged disjoint intervals of ``runs`` inside ``[lo, hi)`` in O(K) memory."""
    if not runs or hi <= lo:
        return
    bases = [r.base_byte for r in runs]
    strides = [r.stride_bytes for r in runs]
    rbs = [r.row_bytes for r in runs]
    secs = [r.sector_bytes for r in runs]
    steps: list[int] = []
    k_his: list[int] = []
    cur_los: list[int] = []
    cur_his: list[int] = []
    active: list[int] = []
    for idx, r in enumerate(runs):
        sec, stride, base, rb = secs[idx], strides[idx], bases[idx], rbs[idx]
        if r.rows <= 0 or rb <= 0 or stride <= 0 or hi - 1 < base:
            steps.append(1)
            k_his.append(0)
            cur_los.append(-1)
            cur_his.append(-1)
            continue
        num_lo = lo - sec - base - rb
        k_lo = 0 if num_lo < 0 else (num_lo // stride) + 1
        k_hi = min(r.rows - 1, (hi - 1 - base) // stride)
        c_lo = c_hi = -1
        st = k_lo
        while st <= k_hi:
            addr = base + st * stride
            s_k = (addr // sec) * sec
            e_k = ((addr + rb + sec - 1) // sec) * sec
            cl = lo if s_k < lo else s_k
            ch = hi if e_k > hi else e_k
            st += 1
            if ch > cl:
                c_lo, c_hi = cl, ch
                break
        steps.append(st)
        k_his.append(k_hi)
        cur_los.append(c_lo)
        cur_his.append(c_hi)
        if c_hi > c_lo:
            active.append(idx)
    m_lo = m_hi = -1
    while active:
        best_pos = 0
        best_i = active[0]
        best_lo = cur_los[best_i]
        for pos in range(1, len(active)):
            idx = active[pos]
            val = cur_los[idx]
            if val < best_lo:
                best_lo = val
                best_i = idx
                best_pos = pos
        best_hi = cur_his[best_i]
        if m_hi < 0:
            m_lo, m_hi = best_lo, best_hi
        elif best_lo <= m_hi:
            if best_hi > m_hi:
                m_hi = best_hi
        else:
            yield m_lo, m_hi
            m_lo, m_hi = best_lo, best_hi
        st = steps[best_i]
        kh = k_his[best_i]
        if st <= kh:
            base, stride, rb, sec = bases[best_i], strides[best_i], rbs[best_i], secs[best_i]
            addr = base + st * stride
            s_k = (addr // sec) * sec
            e_k = ((addr + rb + sec - 1) // sec) * sec
            cl = lo if s_k < lo else s_k
            ch = hi if e_k > hi else e_k
            steps[best_i] = st + 1
            if ch > cl:
                cur_los[best_i] = cl
                cur_his[best_i] = ch
            else:
                active.pop(best_pos)
        else:
            active.pop(best_pos)
    if m_hi >= 0:
        yield m_lo, m_hi


def _stream_union_bytes(runs: tuple[PeriodicSectorRun, ...], lo: int, hi: int) -> int:
    """Exact byte measure of the union of ``runs`` in ``[lo, hi)`` with bounded memory."""
    count = len(runs)
    if count == 0 or hi <= lo:
        return 0
    if count == 1:
        return runs[0].overlap_with_interval(lo, hi)
    if count == _PAIR_RUNS:
        a, b = runs
        return a.overlap_with_interval(lo, hi) + b.overlap_with_interval(lo, hi) - _measure_pair_overlap(a, b, lo, hi)
    if count == _TRIPLE_RUNS:
        a, b, c = runs
        singles = a.overlap_with_interval(lo, hi) + b.overlap_with_interval(lo, hi) + c.overlap_with_interval(lo, hi)
        pairs = (
            _measure_pair_overlap(a, b, lo, hi)
            + _measure_pair_overlap(a, c, lo, hi)
            + _measure_pair_overlap(b, c, lo, hi)
        )
        t_lo = max(lo, a._bounds()[0], b._bounds()[0], c._bounds()[0])
        t_hi = min(hi, a._bounds()[1], b._bounds()[1], c._bounds()[1])
        triple = sum(_run_overlap_small_interval(c, u, v) for u, v in _iter_two_run_overlap(a, b, t_lo, t_hi)) if t_hi > t_lo else 0
        return singles - pairs + triple
    return sum(y - x for x, y in _iter_union_stream(runs, lo, hi))


def _stream_intersect_bytes(
    l_cov: bool, l_runs: tuple[PeriodicSectorRun, ...],
    r_cov: bool, r_runs: tuple[PeriodicSectorRun, ...],
    lo: int, hi: int,
) -> int:
    """Exact byte intersection of left and right active run sets in bounded memory."""
    if hi <= lo:
        return 0
    if l_cov:
        return hi - lo if r_cov else _stream_union_bytes(r_runs, lo, hi)
    if r_cov:
        return _stream_union_bytes(l_runs, lo, hi)
    if not l_runs or not r_runs:
        return 0
    if len(l_runs) == 1 and len(r_runs) == 1:
        return _measure_pair_overlap(l_runs[0], r_runs[0], lo, hi)
    l_iter = _iter_union_stream(l_runs, lo, hi)
    r_iter = _iter_union_stream(r_runs, lo, hi)
    l_iv = next(l_iter, None)
    r_iv = next(r_iter, None)
    total = 0
    while l_iv is not None and r_iv is not None:
        o_lo = l_iv[0] if l_iv[0] >= r_iv[0] else r_iv[0]
        o_hi = l_iv[1] if l_iv[1] <= r_iv[1] else r_iv[1]
        if o_hi > o_lo:
            total += o_hi - o_lo
        if l_iv[1] <= r_iv[1]:
            l_iv = next(l_iter, None)
        else:
            r_iv = next(r_iter, None)
    return total
