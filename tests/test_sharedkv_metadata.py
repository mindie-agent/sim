"""Independent scalar source allocator and continuous-owner invariants."""
import tracemalloc

import pytest

from mindie_sim.prediction.sharedkv_metadata import query_partitions


def _scalar_costs(query, context, window, ratio, topk, heads):
    rows = []
    for q in range(query):
        blocks = []
        for size in (min(context + q + 1, window),
                     min((context + q + 1) // ratio, topk)):
            for offset in range(0, size, 512):
                length = min(512, size - offset)
                blocks.append(6 * ((heads + 15) // 16) + 10 * ((length + 63) // 64))
        rows.append((sum(blocks), blocks[-1]))
    return rows


def _scalar_source_partition(rows, cores):
    # Direct scalar interpretation of AssignByBatch, AssignByRow and
    # IsWithinTolerance at source revision 175ef4d49; no cost-band arithmetic.
    remaining = sum(cost for cost, _ in rows)
    maximum = max(cost for cost, _ in rows)
    result, cursor = [], 0
    for core in range(cores):
        if cursor == len(rows):
            break
        limit = max(remaining // (cores - core), maximum)
        start, spent = cursor, 0
        if remaining <= limit + rows[-1][1] // 2:
            result.append((start, len(rows)))
            break
        while cursor < len(rows):
            cost, tail = rows[cursor]
            if spent + cost > limit + tail // 2:
                break
            spent += cost
            cursor += 1
        remaining -= spent
        result.append((start, cursor))
    return tuple(result)


def test_source_costs_assign_unequal_complete_query_ranges():
    # Three rows cost 16, five cost 32, total 208. The successive limits are
    # 52, 53, 64, 64 and each row's final block gives tolerance 8.
    assert query_partitions(8, 0, 128, 4, 512, 8, 4) == ((0, 3), (3, 4), (4, 6), (6, 8))
    # With surplus physical cores the maximum-row-cost floor coalesces the
    # first two cheap rows; those unused cores must not initialize or execute.
    assert query_partitions(8, 0, 128, 4, 512, 8, 20) == (
        (0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8))


@pytest.mark.parametrize("query,context,window,ratio,topk,heads,cores", [
    (3, 0, 128, 128, 512, 4, 20),
    (65, 0, 65, 4, 512, 8, 4),
    (257, 0, 512, 4, 512, 12, 20),
    (65, 2047, 128, 4, 1024, 16, 7),
    (129, 4095, 128, 4, 1024, 8, 20),
    (513, 0, 1, 128, 512, 8, 3),
    (513, 8191, 128, 128, 512, 8, 20),
    (4096, 0, 128, 4, 512, 8, 20),
    (1024, 0, 128, 4, 1024, 8, 20),
    (1024, 3072, 128, 4, 1024, 8, 20),
])
def test_partition_matches_scalar_metadata_with_coverage_and_cost_boundaries(
        query, context, window, ratio, topk, heads, cores):
    rows = _scalar_costs(query, context, window, ratio, topk, heads)
    partitions = query_partitions(query, context, window, ratio, topk, heads, cores)
    assert partitions == _scalar_source_partition(rows, cores)
    assert partitions[0][0] == 0 and partitions[-1][1] == query
    assert all(start < stop for start, stop in partitions)
    assert all(left[1] == right[0] for left, right in zip(partitions, partitions[1:]))
    assert len(partitions) <= cores
    remaining = sum(cost for cost, _ in rows)
    maximum = max(cost for cost, _ in rows)
    for core, (start, stop) in enumerate(partitions):
        limit = max(remaining // (cores - core), maximum)
        work = sum(cost for cost, _ in rows[start:stop])
        assert work <= limit + rows[stop - 1][1] // 2
        if stop < query:
            assert work + rows[stop][0] > limit + rows[stop][1] // 2
        remaining -= work
    assert remaining == 0


def test_large_query_partition_retains_only_cost_bands_and_owner_ranges():
    query_partitions.cache_clear()
    tracemalloc.start()
    try:
        partitions = query_partitions(10**12, 0, 512, 128, 1024, 8, 32)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert len(partitions) == 32
    assert sum(stop - start for start, stop in partitions) == 10**12
    assert peak < 64 * 1024
