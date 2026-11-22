"""Typed cache access geometry and compact transaction summaries.

[Role] Describe an access by offset/stride/page/packing/range and derive
  transactions without expanding one object per byte or per row.
[Invariants] A set of touched positions is a union of compact spans:
  ``start + index * stride`` covering ``width`` units for ``count`` indices.
  Union/intersection/subtraction/take/drop operate on those spans, so a
  contiguous run and a repeated strided range use the same arithmetic. Every
  projection (bytes, transaction lines, coverage, RMW, pages) comes from that
  one set. Physical row bytes are post-packing and are not rescaled again.
[Timeouts & Locks] Constant or period-bounded arithmetic, no I/O or locks.
  Explicit enumeration is bounded; an unbounded request is rejected rather
  than silently rounded to a span.
[Side-Effect Policy] Immutable values only.
"""
from dataclasses import dataclass
from heapq import heappop, heappush
from math import gcd

from mindie_sim.schema.common import ModelingError


Range = tuple[int, int]
# Range enumeration is a reporting convenience, never the internal set
# representation. Above this many runs the compact spans remain authoritative.
_ENUM_LIMIT = 1 << 16
_PERIODIC_ROW_SCAN_LIMIT = 256
_PERIODIC_ROW_SCAN_COST_RATIO = 8
_PERIODIC_PHASE_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class Span:
    """One compact set: ``count`` runs of ``width`` at ``stride`` spacing.

    A contiguous range is the single-run form ``Span(start, count, count, 1)``.
    ``width == stride`` or ``count == 1`` denotes a contiguous interval.
    """

    start: int
    width: int
    stride: int
    count: int

    def __post_init__(self):
        for name in ("start", "width", "stride", "count"):
            value = getattr(self, name)
            if type(value) is not int:
                raise ModelingError(f"cache_span_requires_integer:{name}")
        if self.start < 0:
            raise ModelingError("cache_span_start_must_be_nonnegative")
        if self.width <= 0 or self.count <= 0:
            raise ModelingError("cache_span_width_and_count_must_be_positive")
        if self.stride < self.width:
            raise ModelingError("cache_span_stride_must_not_be_below_width")

    @property
    def end(self) -> int:
        return self.start + (self.count - 1) * self.stride + self.width

    @property
    def total(self) -> int:
        return self.width * self.count

    @property
    def contiguous(self) -> bool:
        return self.count == 1 or self.width == self.stride

    def to_tuple(self):
        return (self.start, self.width, self.stride, self.count)

    def to_dict(self) -> dict:
        return {"start": self.start, "width": self.width,
                "stride": self.stride, "count": self.count}


def span(start: int, count: int) -> Span:
    """Single contiguous run."""
    if count <= 0:
        raise ModelingError("cache_span_count_must_be_positive")
    return Span(int(start), int(count), int(count), 1)


def _canonical(item: Span) -> Span:
    if item.count == 1:
        return span(item.start, item.width)
    if item.width == item.stride:
        return span(item.start, item.total)
    return item


def range_count(spans) -> int:
    return sum(item.total for item in spans)


def normalize_spans(spans):
    """Return disjoint, ordered spans covering exactly the union of ``spans``."""
    items = [_canonical(item) for item in spans if item.count > 0]
    items.sort(key=lambda item: (item.start, item.end, item.stride))
    result = []
    for item in items:
        remaining = [item]
        for prior in result:
            if not remaining:
                break
            peeled = []
            for piece in remaining:
                peeled.extend(subtract_span(piece, prior))
            remaining = peeled
        result.extend(remaining)
    result.sort(key=lambda item: (item.start, item.end, item.stride))
    merged = []
    for item in result:
        if merged:
            prior = merged[-1]
            if prior.contiguous and item.contiguous and item.start <= prior.end:
                merged[-1] = span(prior.start, max(prior.end, item.end)
                                  - prior.start)
                continue
            if (not prior.contiguous and not item.contiguous
                    and prior.width == item.width
                    and prior.stride == item.stride
                    and item.start == prior.start + prior.count * prior.stride):
                merged[-1] = Span(prior.start, prior.width, prior.stride,
                                  prior.count + item.count)
                continue
        merged.append(item)
    return tuple(merged)


def union_spans(*sets):
    return normalize_spans(item for group in sets for item in group)


def _interval_span(start, end):
    return (span(start, end - start),) if end > start else ()


def _intersect_interval_with_repeat(interval, item):
    lo, hi = interval
    first = -((-(lo - (item.start + item.width) + 1)) // item.stride)
    last = (hi - 1 - item.start) // item.stride
    first = max(0, first)
    last = min(item.count - 1, last)
    if first > last:
        return ()
    pieces = []
    row = item.start + first * item.stride
    left, right = max(lo, row), min(hi, row + item.width)
    if right > left:
        pieces.append(span(left, right - left))
    if last > first + 1:
        pieces.append(Span(row + item.stride, item.width, item.stride,
                           last - first - 1))
    if last > first:
        row = item.start + last * item.stride
        left, right = max(lo, row), min(hi, row + item.width)
        if right > left:
            pieces.append(span(left, right - left))
    return normalize_spans(pieces)


def _intersect_point_progressions(left: Span, right: Span):
    """Exact intersection of two unit-width arithmetic progressions."""
    d1, d2 = left.stride, right.stride
    divisor = gcd(d1, d2)
    delta = right.start - left.start
    if delta % divisor:
        return ()
    period = d1 // divisor * d2
    modulus = d2 // divisor
    if modulus == 1:
        residue = 0
    else:
        residue = ((delta // divisor)
                   * pow(d1 // divisor, -1, modulus)) % modulus
    need = max(left.start, right.start)
    start = left.start + residue * d1
    if start < need:
        steps = (need - start + period - 1) // period
        start += steps * period
    left_index = (start - left.start) // d1
    right_index = (start - right.start) // d2
    if left_index >= left.count or right_index >= right.count:
        return ()
    last = min(left.start + (left.count - 1) * d1,
               right.start + (right.count - 1) * d2)
    count = (last - start) // period + 1
    return (Span(start, 1, period, count),)


def _ceil_div(value: int, divisor: int) -> int:
    return -((-value) // divisor)


def _intersect_same_stride(left: Span, right: Span):
    """Exact intersection when both progressions share one step.

    A phase shift can make every row overlap (or touch nothing). The overlap
    within a row is constant for one index difference, so the result stays
    compact even when the row count is large.
    """
    stride = left.stride
    delta = right.start - left.start
    # With e = Right - Left, intervals overlap iff
    # -right.width < e < left.width.
    first_shift = (-right.width - delta) // stride + 1
    last_shift = _ceil_div(left.width - delta, stride) - 1
    if first_shift > last_shift:
        return ()
    pieces = []
    for shift in range(first_shift, last_shift + 1):
        first = max(0, -shift)
        last = min(left.count - 1, right.count - 1 - shift)
        if first > last:
            continue
        offset = delta + shift * stride
        if offset >= 0:
            start = left.start + first * stride + offset
            width = min(left.width - offset, right.width)
        else:
            start = left.start + first * stride
            width = min(left.width, right.width + offset)
        if width <= 0:
            continue
        count = last - first + 1
        pieces.append(span(start, width) if count == 1 else
                      Span(start, width, stride, count))
    return normalize_spans(pieces)


def _intersect_periodic_rows(small: Span, large: Span):
    """Exact periodic intersection by iterating only the smaller row set.

    Two rows are cheap even when the relative-phase candidate count
    ``(width_left+width_right)/gcd`` is enormous.  This is the bounded form
    used when the finite row pairs are fewer than the phase candidates.
    """
    pieces = []
    for index in range(small.count):
        row = small.start + index * small.stride
        pieces.extend(_intersect_interval_with_repeat(
            (row, row + small.width), large))
    return normalize_spans(pieces)


def _intersect_periodic(left: Span, right: Span):
    """Exact intersection of two interval progressions with different steps.

    For a fixed ``t = i*left.stride - j*right.stride`` the relative row offset
    is constant. The admissible ``t`` values number about
    ``(left.width + right.width) / gcd(steps)`` rather than the row counts, and
    solutions for one ``t`` form one strided span. No row or LCM expansion is
    materialised here.

    When the smaller side has only a few rows, iterating those finite row
    pairs is cheaper than the phase enumeration and remains exact.  For dense
    progressions the phase list is consumed in bounded chunks so a very large
    ``width/gcd`` cannot create an unbounded intermediate list.
    """
    a, b = left.stride, right.stride
    divisor = gcd(a, b)
    a_reduced, b_reduced = a // divisor, b // divisor
    delta = right.start - left.start
    low = delta - left.width + 1
    high = delta + right.width - 1
    first_t = low + (-low) % divisor
    candidate_count = (high - first_t) // divisor + 1
    if candidate_count <= 0:
        return ()
    small, large = ((left, right) if left.count <= right.count
                    else (right, left))
    # The finite row pairs are the real bound for short descriptions.  Keep
    # the old phase arithmetic for dense progressions; only its temporary
    # work is chunked, never the result semantics.
    if (small.count <= _PERIODIC_ROW_SCAN_LIMIT
            or small.count * _PERIODIC_ROW_SCAN_COST_RATIO < candidate_count):
        return _intersect_periodic_rows(small, large)
    if b_reduced == 1:
        inverse = 0
    else:
        inverse = pow(a_reduced % b_reduced, -1, b_reduced)
    pieces = []
    chunk = _PERIODIC_PHASE_CHUNK_SIZE
    for chunk_start in range(first_t, high + 1, divisor * chunk):
        chunk_end = min(high + 1, chunk_start + divisor * chunk)
        for t in range(chunk_start, chunk_end, divisor):
            q = t // divisor
            i0 = (q * inverse) % b_reduced if b_reduced > 1 else 0
            j0 = (i0 * a_reduced - q) // b_reduced
            first = max(_ceil_div(-i0, b_reduced),
                        _ceil_div(-j0, a_reduced))
            last = min((left.count - 1 - i0) // b_reduced,
                       (right.count - 1 - j0) // a_reduced)
            if first > last:
                continue
            relative = delta - t
            row_start = left.start + (i0 + first * b_reduced) * a
            if relative >= 0:
                start = row_start + relative
                width = min(left.width - relative, right.width)
            else:
                start = row_start
                width = min(left.width, right.width + relative)
            if width <= 0:
                continue
            count = last - first + 1
            step = a * b_reduced
            pieces.append(span(start, width) if count == 1 else
                          Span(start, width, step, count))
        pieces = list(normalize_spans(pieces))
    return normalize_spans(pieces)


def intersect_span(left: Span, right: Span):
    if left.contiguous and right.contiguous:
        start = max(left.start, right.start)
        end = min(left.end, right.end)
        return _interval_span(start, end)
    if left.contiguous:
        return _intersect_interval_with_repeat((left.start, left.end), right)
    if right.contiguous:
        return _intersect_interval_with_repeat((right.start, right.end), left)
    if left.stride == right.stride:
        return _intersect_same_stride(left, right)
    if left.width == 1 and right.width == 1:
        return _intersect_point_progressions(left, right)
    return _intersect_periodic(left, right)


def intersect_spans(left, right):
    left, right = normalize_spans(left), normalize_spans(right)
    result = []
    # Sparse spans have holes, so ``end`` is not a sweep cursor: a span may
    # end beyond a later span while its last row starts after it. Pairwise
    # intersection keeps the compact arithmetic exact; normalized sets are
    # small in this cache model.
    for a in left:
        for b in right:
            if a.start < b.end and b.start < a.end:
                result.extend(intersect_span(a, b))
    return normalize_spans(result)


def _subtract_interval_from_repeat(item: Span, lo: int, hi: int):
    first = -((-(lo - (item.start + item.width) + 1)) // item.stride)
    last = (hi - 1 - item.start) // item.stride
    first = max(0, first)
    last = min(item.count - 1, last)
    if first > last:
        return (item,)
    pieces = []
    if first > 0:
        pieces.append(Span(item.start, item.width, item.stride, first))
    row = item.start + first * item.stride
    left = min(lo, row + item.width)
    if left > row:
        pieces.append(span(row, left - row))
    # Rows strictly between the clipped head and tail are fully removed.
    row = item.start + last * item.stride
    right = max(hi, row)
    if right < row + item.width:
        pieces.append(span(right, row + item.width - right))
    if last + 1 < item.count:
        pieces.append(Span(item.start + (last + 1) * item.stride, item.width,
                           item.stride, item.count - last - 1))
    return normalize_spans(pieces)


def _subtract_interval_from_run(item: Span, lo: int, hi: int):
    pieces = []
    if item.start < lo:
        pieces.append(span(item.start, min(item.end, lo) - item.start))
    if item.end > hi:
        pieces.append(span(max(item.start, hi), item.end - max(item.start, hi)))
    return tuple(pieces)


def _subtract_periodic_from_span(left: Span, right: Span):
    """Exact difference using the complement pieces of the removed repeat.

    The complement is the head before the first removed row, the gaps between
    consecutive removed rows, and the tail after the last row. Each gap set is
    itself one compact span, so this uses the exact intersection arithmetic
    instead of iterating one row (or one shortened remainder) at a time.
    """
    pieces = []
    if right.start > left.start:
        head_end = min(right.start, left.end)
        if head_end > left.start:
            pieces.extend(intersect_span(
                left, span(left.start, head_end - left.start)))
    if right.count > 1 and right.stride > right.width:
        gaps = Span(right.start + right.width, right.stride - right.width,
                    right.stride, right.count - 1)
        pieces.extend(intersect_span(left, gaps))
    if right.end < left.end:
        tail_start = max(right.end, left.start)
        if tail_start < left.end:
            pieces.extend(intersect_span(
                left, span(tail_start, left.end - tail_start)))
    return normalize_spans(pieces)


def subtract_span(left: Span, right: Span):
    """Return ``left \\ right`` as compact spans."""
    if left.contiguous and right.contiguous:
        return _subtract_interval_from_run(left, right.start, right.end)
    if right.contiguous:
        return _subtract_interval_from_repeat(left, right.start, right.end)
    return _subtract_periodic_from_span(left, right)


def subtract_spans(left, right):
    result = []
    right = normalize_spans(right)
    for item in normalize_spans(left):
        current = (item,)
        for removed in right:
            if not current:
                break
            updated = []
            for piece in current:
                updated.extend(subtract_span(piece, removed))
            current = tuple(updated)
        result.extend(current)
    return normalize_spans(result)


def ranges_of(spans, limit: int | None = None):
    """Enumerate compact spans to ``(start, count)`` runs, with a bound."""
    limit = _ENUM_LIMIT if limit is None else int(limit)
    result = []
    for item in normalize_spans(spans):
        if item.contiguous:
            result.append((item.start, item.total))
            continue
        if len(result) + item.count > limit:
            raise ModelingError("cache_span_enumeration_exceeds_bound")
        for index in range(item.count):
            result.append((item.start + index * item.stride, item.width))
    merged = []
    for start, count in sorted(result):
        if merged and start <= merged[-1][0] + merged[-1][1]:
            merged[-1] = (merged[-1][0],
                          max(merged[-1][0] + merged[-1][1], start + count)
                          - merged[-1][0])
        else:
            merged.append((start, count))
    return tuple(merged)


def try_ranges_of(spans, limit: int | None = None):
    try:
        return ranges_of(spans, limit)
    except ModelingError as exc:
        if str(exc) == "cache_span_enumeration_exceeds_bound":
            return None
        raise


def take_first_spans(spans, limit):
    """Keep the first ``limit`` covered units in spatial order.

    A compact repeat can enclose another disjoint span in its gaps, so span
    start order is not unit order. A small priority queue streams the next row
    of each span; complete single-span prefixes are consumed in one step.
    """
    remaining = max(0, int(limit))
    items = normalize_spans(spans)
    if remaining <= 0 or not items:
        return ()
    if remaining >= range_count(items):
        return items
    heap = []
    for order, item in enumerate(items):
        heappush(heap, (item.start, order, item, 0))
    result = []
    while remaining > 0 and heap:
        start, order, item, index = heappop(heap)
        rows_left = item.count - index
        if item.contiguous:
            take = min(remaining, item.width)
            result.append(span(start, take))
            remaining -= take
            continue
        next_start = heap[0][0] if heap else None
        if next_start is None:
            rows_bound = rows_left
        else:
            rows_bound = (next_start - start + item.stride - 1) // item.stride
            rows_bound = max(0, min(rows_left, rows_bound))
        whole_rows = min(rows_bound, remaining // item.width)
        if whole_rows:
            result.append(Span(start, item.width, item.stride, whole_rows))
            remaining -= whole_rows * item.width
            index += whole_rows
            if index < item.count:
                heappush(heap, (item.start + index * item.stride, order,
                                item, index))
            continue
        take = min(remaining, item.width)
        result.append(span(start, take))
        remaining -= take
        index += 1
        if remaining and index < item.count:
            heappush(heap, (item.start + index * item.stride, order, item,
                            index))
    return normalize_spans(result)


def drop_last_spans(spans, limit):
    """Drop the last ``limit`` covered units; return (kept, dropped)."""
    items = normalize_spans(spans)
    kept = take_first_spans(items, max(0, range_count(items) - int(limit)))
    dropped = subtract_spans(items, kept)
    return kept, dropped


def lines_of_byte_spans(spans, line: int):
    """Project byte spans to the transaction lines they touch.

    Alignment/stride phase is exploited directly. For a non-aligned repeat the
    line shape repeats with the phase period ``line / gcd(stride, line)``, so
    the projection has at most that many segments, independent of row count.
    """
    if type(line) is not int or line < 1:
        raise ModelingError("cache_line_bytes_must_be_positive_integer")
    result = []
    for item in normalize_spans(spans):
        if item.contiguous:
            first = item.start // line
            last = (item.end - 1) // line
            result.append(span(first, last - first + 1))
            continue
        if item.start % line == 0 and item.width % line == 0 \
                and item.stride % line == 0:
            result.append(Span(item.start // line, item.width // line,
                               item.stride // line, item.count))
            continue
        period = line // gcd(item.stride, line)
        for residue in range(min(period, item.count)):
            row_start = item.start + residue * item.stride
            first = row_start // line
            last = (row_start + item.width - 1) // line
            count = (item.count - 1 - residue) // period + 1
            width = last - first + 1
            step = (item.stride * period) // line
            if step <= width:
                result.append(span(first, (count - 1) * step + width))
            else:
                result.append(Span(first, width, step, count))
    return normalize_spans(result)


def byte_spans_of_line_spans(spans, line: int):
    if type(line) is not int or line < 1:
        raise ModelingError("cache_line_bytes_must_be_positive_integer")
    return normalize_spans(
        Span(item.start * line, item.width * line, item.stride * line,
             item.count) for item in normalize_spans(spans))


def full_line_spans(byte_spans, line: int):
    """Lines completely covered by the byte union, as transaction spans.

    A line is incomplete exactly when it meets one of the gaps between the
    first and last touched byte. The gap set is the contiguous envelope minus
    the normalised union, so this remains exact for lines completed jointly by
    several spans and never materialises one object per physical line.
    """
    if type(line) is not int or line < 1:
        raise ModelingError("cache_line_bytes_must_be_positive_integer")
    items = normalize_spans(byte_spans)
    if not items:
        return ()
    first_line = items[0].start // line
    # Spans are ordered by start, not by their true maximum end: a short span
    # starting later may sit wholly inside the gaps of a longer earlier span.
    # Use the real envelope end so lines beyond the last start are still
    # checked for gaps.
    last_line = (max(item.end for item in items) - 1) // line
    envelope = span(first_line * line,
                    (last_line - first_line + 1) * line)
    gaps = subtract_spans((envelope,), items)
    touched = lines_of_byte_spans(items, line)
    nonfull = lines_of_byte_spans(gaps, line)
    return normalize_spans(subtract_spans(touched, nonfull))


@dataclass(frozen=True)
class TransactionSummary:
    """Compact transaction view of one typed access range."""

    line_bytes: int
    first_line: int
    line_count: int
    logical_bytes: int
    partial_first: bool
    partial_last: bool
    rmw_read_lines: int
    page_count: int
    repeated: bool
    rows: int
    row_bytes: int
    stride_bytes: int
    packing: int
    # Compact authoritative sets. ``line_ranges``/``byte_ranges`` are the
    # legacy enumeration and are only populated while bounded.
    line_segments: tuple[Span, ...] = ()
    byte_segments: tuple[Span, ...] = ()
    coverage_bytes: int = 0
    gm_bytes: int = 0
    line_ranges: tuple[Range, ...] = ()
    byte_ranges: tuple[Range, ...] = ()
    ranges_complete: bool = False

    def to_dict(self) -> dict:
        return {
            "line_bytes": self.line_bytes, "first_line": self.first_line,
            "line_count": self.line_count, "logical_bytes": self.logical_bytes,
            "partial_first": self.partial_first,
            "partial_last": self.partial_last,
            "rmw_read_lines": self.rmw_read_lines,
            "page_count": self.page_count, "repeated": self.repeated,
            "rows": self.rows, "row_bytes": self.row_bytes,
            "stride_bytes": self.stride_bytes, "packing": self.packing,
            "line_segments": [item.to_tuple() for item in self.line_segments],
            "byte_segments": [item.to_tuple() for item in self.byte_segments],
            "line_ranges": [list(item) for item in self.line_ranges],
            "byte_ranges": [list(item) for item in self.byte_ranges],
            "ranges_complete": self.ranges_complete,
            "coverage_bytes": self.coverage_bytes,
            "gm_bytes": self.gm_bytes,
        }


@dataclass(frozen=True)
class CacheAccessGeometry:
    """Typed access range: offset/stride, pages, packing and partial lines.

    ``row_bytes`` is the physical post-packing row width supplied by the
    representation; ``packing`` is retained as a declared protocol factor and
    is not used to rescale that width a second time. ``stride_bytes=0`` means
    contiguous rows. ``partial_line="read_modify_write"`` adds the necessary
    partial-line reads; ``"masked"`` charges write lines only.
    """

    offset_bytes: int = 0
    rows: int = 1
    row_bytes: int = 0
    stride_bytes: int = 0
    page_bytes: int = 0
    packing: int = 1
    partial_line: str = "masked"

    def __post_init__(self):
        for name in ("offset_bytes", "rows", "row_bytes", "stride_bytes",
                     "page_bytes", "packing"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ModelingError(f"cache_access_geometry_invalid:{name}")
        if self.packing < 1:
            raise ModelingError("cache_access_geometry_packing_must_be_positive")
        if self.partial_line not in {"masked", "read_modify_write"}:
            raise ModelingError("cache_access_geometry_unknown_partial_line_protocol")
        if self.rows and self.row_bytes == 0:
            raise ModelingError("cache_access_geometry_row_bytes_must_be_positive")

    @property
    def logical_bytes(self) -> int:
        return self.rows * self.row_bytes

    @property
    def stride(self) -> int:
        return self.stride_bytes or self.row_bytes

    def summary(self, line_bytes: int, *, write: bool = False,
                transfer_bytes: int = 1) -> TransactionSummary:
        if type(line_bytes) is not int or line_bytes < 1:
            raise ModelingError("cache_access_geometry_line_bytes_must_be_positive")
        rows, row_bytes, stride = self.rows, self.row_bytes, self.stride
        if not rows or not row_bytes:
            return TransactionSummary(line_bytes, 0, 0, 0, False, False, 0, 0,
                                      False, rows, row_bytes, stride,
                                      self.packing)
        start = self.offset_bytes
        end = start + (rows - 1) * stride + row_bytes
        if stride < row_bytes:
            byte_segments = (span(start, end - start),)
        elif rows == 1:
            byte_segments = (span(start, row_bytes),)
        else:
            byte_segments = (Span(start, row_bytes, stride, rows),)
        byte_segments = normalize_spans(byte_segments)
        line_segments = lines_of_byte_spans(byte_segments, line_bytes)
        line_count = range_count(line_segments)
        coverage_bytes = range_count(byte_segments)
        first_line = line_segments[0].start
        last_line = max(item.end for item in line_segments) - 1

        def covered_in(line_no):
            return range_count(intersect_spans(
                byte_segments, (span(line_no * line_bytes, line_bytes),)))

        partial_first = covered_in(first_line) < line_bytes
        partial_last = covered_in(last_line) < line_bytes
        rmw_read_lines = 0
        if write and self.partial_line == "read_modify_write":
            rmw_read_lines = range_count(subtract_spans(
                line_segments, full_line_spans(byte_segments, line_bytes)))
        pages = 0
        if self.page_bytes:
            pages = range_count(lines_of_byte_spans(
                byte_segments, self.page_bytes))
        gm = (rows * (((row_bytes + transfer_bytes - 1) // transfer_bytes)
                      * transfer_bytes) if transfer_bytes > 0
              else self.logical_bytes)
        line_ranges = try_ranges_of(line_segments, _ENUM_LIMIT)
        byte_ranges = try_ranges_of(byte_segments, _ENUM_LIMIT)
        return TransactionSummary(
            line_bytes, first_line, line_count, self.logical_bytes,
            partial_first, partial_last, rmw_read_lines, pages,
            rows > 1 and stride != row_bytes, rows, row_bytes, stride,
            self.packing, line_segments, byte_segments, coverage_bytes, gm,
            () if line_ranges is None else line_ranges,
            () if byte_ranges is None else byte_ranges,
            line_ranges is not None and byte_ranges is not None)


__all__ = [
    "CacheAccessGeometry", "Span", "TransactionSummary", "byte_spans_of_line_spans",
    "drop_last_spans", "full_line_spans", "intersect_span", "intersect_spans",
    "lines_of_byte_spans", "normalize_spans", "range_count", "ranges_of",
    "span", "subtract_span", "subtract_spans", "take_first_spans",
    "try_ranges_of", "union_spans",
]
