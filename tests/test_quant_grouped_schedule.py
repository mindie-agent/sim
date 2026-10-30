"""Independent source-schedule geometry checks for quant_grouped_schedule."""
from dataclasses import astuple, replace
import itertools
import json
from pathlib import Path
import tracemalloc

import pytest

from mindie_sim.prediction.operators.quant_grouped_schedule import (
    QuantGroupedTiling,
    plan_quant_grouped,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/calls/quantized-grouped-source-geometry.json"


def _tiles(schedule, owner):
    return list(schedule.iter_owner_tiles(owner))


def _all_tiles(schedule):
    return [
        tile
        for owner in range(schedule.tiling.used_cores)
        for tile in schedule.iter_owner_tiles(owner)
    ]


def test_four_row_serpentine_uses_literal_owner_rectangles():
    schedule = plan_quant_grouped(
        (128,), 64, 16, QuantGroupedTiling(16, 32, 32, 16)
    )
    expected = [
        (0, 0), (16, 0), (32, 0), (48, 0),
        (0, 32), (16, 32), (32, 32), (48, 32),
        (64, 32), (80, 32), (96, 32), (112, 32),
        (64, 0), (80, 0), (96, 0), (112, 0),
    ]
    observed = []
    for owner in range(16):
        tiles = _tiles(schedule, owner)
        assert len(tiles) == 1
        observed.append((tiles[0].m_offset, tiles[0].n_offset))
    assert observed == expected


def test_rotation_wrap_and_zero_groups_preserve_owner_order_and_banks():
    schedule = plan_quant_grouped(
        (48, 0, 96, 0), 32, 16, QuantGroupedTiling(16, 32, 32, 8)
    )
    # group 0 fills owners 0..2; the zero group does not rotate;
    # group 2 starts at owner 3 and wraps through owner 0.
    expected = {
        0: [(0, 0), (2, 80)],
        1: [(0, 16)],
        2: [(0, 32)],
        3: [(2, 0)],
        4: [(2, 16)],
        5: [(2, 32)],
        6: [(2, 48)],
        7: [(2, 64)],
    }
    for owner, rectangles in expected.items():
        assert [
            (tile.group, tile.m_offset) for tile in _tiles(schedule, owner)
        ] == rectangles

    # Group 2 owns source bank 2 even though group 1 is empty.
    group2 = [tile for tile in _tiles(schedule, 3) if tile.group == 2]
    assert len(group2) == 1
    tile = group2[0]
    assert tile.weight_offset == 2 * 32 * 16
    assert tile.column_scale_offset == 2 * 32 * 4
    assert tile.x_offset == (48 + 0) * 16
    assert tile.output_offset == (48 * 32 + 0) * 2
    assert tile.row_scale_offset == 48 * 4
    assert all(tile.group != 1 for tile in _all_tiles(schedule))


def test_trailing_zero_group_disables_split_of_last_nonempty_expert():
    tiling = QuantGroupedTiling(32, 32, 32, 8)
    split = plan_quant_grouped((64,), 64, 16, tiling)
    trailing = plan_quant_grouped((64, 0), 64, 16, tiling)

    split_tiles = _all_tiles(split)
    trailing_tiles = _all_tiles(trailing)
    assert len(split_tiles) == 8
    assert len(trailing_tiles) == 4
    assert {(tile.m_offset, tile.n_offset, tile.m, tile.n) for tile in split_tiles} == {
        (0, 0, 16, 32), (16, 0, 16, 32),
        (32, 0, 16, 32), (48, 0, 16, 32),
        (0, 32, 16, 32), (16, 32, 16, 32),
        (32, 32, 16, 32), (48, 32, 16, 32),
    }
    assert {(tile.m_offset, tile.n_offset, tile.m, tile.n) for tile in trailing_tiles} == {
        (0, 0, 32, 32), (32, 0, 32, 32),
        (0, 32, 32, 32), (32, 32, 32, 32),
    }
    assert trailing.summary()["geometry"]["last_group_splits"] is False
    assert trailing.summary()["geometry"]["trailing_empty_groups"] == 1


def test_n_tail_split_uses_source_128_granule_not_nz_32_storage_block():
    schedule = plan_quant_grouped(
        (16,), 256, 16, QuantGroupedTiling(16, 256, 32, 8)
    )
    expected = [
        (0, 0, 16, 128, 0),
        (0, 0, 16, 128, 128),
    ]
    for owner, (group, ordinal, m, n, n_offset) in enumerate(expected):
        tiles = _tiles(schedule, owner)
        assert [(tile.group, tile.ordinal, tile.m, tile.n, tile.n_offset)
                for tile in tiles] == [(group, ordinal, m, n, n_offset)]
        tile = tiles[0]
        assert tile.weight_offset == n_offset * 16
        assert list(schedule.iter_weight_spans(tile)) == [
            (n_offset * 16, 128 * 16)
        ]
    for owner in range(2, 8):
        assert _tiles(schedule, owner) == []
    small = plan_quant_grouped(
        (16,), 96, 16, QuantGroupedTiling(16, 96, 32, 6)
    )
    assert [(tile.m, tile.n) for tile in _all_tiles(small)] == [(16, 96)]
    assert small.owner_tile_counts == (1, 0, 0, 0, 0, 0)


def test_empty_final_m_split_owner_emits_no_tile_and_keeps_ordinal():
    # Last base tile (289 x 32) is split into 18 M rectangles of 17 rows;
    # offset 289 is empty in the 18th owner, matching source early return.
    schedule = plan_quant_grouped(
        (289,), 608, 16, QuantGroupedTiling(304, 32, 32, 18)
    )
    owner17 = _tiles(schedule, 17)
    assert [(tile.ordinal, tile.m_offset, tile.n_offset, tile.m, tile.n)
            for tile in owner17] == [(0, 0, 544, 289, 32)]

    for owner in range(17):
        tiles = _tiles(schedule, owner)
        assert len(tiles) == 2
        assert tiles[-1].m_offset == owner * 17
        assert tiles[-1].m == 17
        assert tiles[-1].ordinal == 1
    assert schedule.summary()["geometry"]["last_group_splits"] is True


def test_example_geometry_matches_literal_owner_tile_declaration():
    declaration = json.loads(EXAMPLE.read_text())
    arguments = declaration["geometry"]
    tiling = QuantGroupedTiling(**arguments["tiling"])
    schedule = plan_quant_grouped(
        tuple(arguments["expert_rows"]),
        arguments["n"],
        arguments["k"],
        tiling,
    )
    expected = declaration["expected"]
    assert schedule.total_m == expected["total_m"]
    assert len(_all_tiles(schedule)) == expected["tile_count"]
    for owner, tiles in expected["owner_tiles"].items():
        assert [list(astuple(tile)) for tile in _tiles(schedule, int(owner))] == tiles


def test_nz_weight_offsets_against_explicit_tiny_block_storage_enumeration():
    expert_rows, n, k = (16, 0, 16), 64, 32
    tiling = QuantGroupedTiling(16, 32, 32, 4)
    schedule = plan_quant_grouped(expert_rows, n, k, tiling)
    n_blocks, k_blocks = n // 32, k // 16

    # Physical [E, N/32, K/16, 16, 32]: fill every logical weight element with
    # a unique byte.  Group 1 is an untouched bank.
    storage = bytearray(len(expert_rows) * n * k)
    for group in range(len(expert_rows)):
        for n_block in range(n_blocks):
            for k_block in range(k_blocks):
                for inner_k in range(16):
                    for inner_n in range(32):
                        kk = k_block * 16 + inner_k
                        nn = n_block * 32 + inner_n
                        value = (group * n * k + nn * k + kk) % 251 + 1
                        index = (
                            ((group * n_blocks + n_block) * k_blocks + k_block)
                            * 16 * 32
                            + inner_k * 32
                            + inner_n
                        )
                        storage[index] = value

    for tile in _all_tiles(schedule):
        assert tile.n_offset % 32 == 0 and tile.n % 32 == 0
        logical = sorted(
            storage[
                (
                    (tile.group * n_blocks + tile.n_offset // 32) * k_blocks
                    * 16 * 32
                ):(
                    (tile.group * n_blocks + (tile.n_offset + tile.n) // 32)
                    * k_blocks * 16 * 32
                )
            ]
        )
        expected = sorted(
            (tile.group * n * k + nn * k + kk) % 251 + 1
            for kk in range(k)
            for nn in range(tile.n_offset, tile.n_offset + tile.n)
        )
        assert logical == expected
        spans = list(schedule.iter_weight_spans(tile))
        assert spans == [(tile.group * n * k + tile.n_offset * k, tile.n * k)]
        physical = [
            value
            for offset, length in spans
            for value in storage[offset:offset + length]
        ]
        assert sorted(physical) == expected

    tile = _tiles(schedule, 2)[0]
    with pytest.raises(ValueError, match="whole_nz_n_blocks"):
        list(schedule.iter_weight_spans(replace(tile, n_offset=1)))
    with pytest.raises(ValueError, match="unsupported_sub_block_k_offset"):
        list(schedule.iter_weight_spans(tile, k_offset=1))


def test_nz_subpanels_preserve_physical_holes_and_empty_expert_banks():
    schedule = plan_quant_grouped(
        (1, 0, 1, 0), 128, 64, QuantGroupedTiling(16, 128, 32, 4)
    )
    storage = [
        (group, k_block * 16 + inner_k, n_block * 32 + inner_n)
        for group in range(4)
        for n_block in range(4)
        for k_block in range(4)
        for inner_k in range(16)
        for inner_n in range(32)
    ]
    for tile in _all_tiles(schedule):
        for k_offset, k_size in ((0, 16), (16, 32), (48, 16)):
            spans = list(schedule.iter_weight_spans(tile, k_offset, k_size))
            actual = {
                element for offset, size in spans
                for element in storage[offset:offset + size]
            }
            expected = {
                (tile.group, kk, nn)
                for kk in range(k_offset, k_offset + k_size)
                for nn in range(tile.n_offset, tile.n_offset + tile.n)
            }
            assert actual == expected
            assert len(actual) == sum(size for _, size in spans) == tile.n * k_size
            assert len(spans) == tile.n // 32
            assert all(spans[i + 1][0] > spans[i][0] + spans[i][1]
                       for i in range(len(spans) - 1))
    group2 = _tiles(schedule, 1)[0]
    assert group2.group == 2
    assert list(schedule.iter_weight_spans(group2, 16, 32))[0] == (
        2 * 128 * 64 + 16 * 32, 32 * 32)


@pytest.mark.parametrize(
    "rows,n,k,tiling",
    [
        ((64, 32, 16), 96, 32, QuantGroupedTiling(32, 64, 32, 8)),
        ((17, 0, 33, 5), 32, 16, QuantGroupedTiling(16, 32, 32, 4)),
        ((1, 2, 3, 4, 5), 64, 48, QuantGroupedTiling(48, 32, 64, 8)),
        ((96,), 64, 16, QuantGroupedTiling(16, 32, 32, 12)),
        ((128,), 64, 16, QuantGroupedTiling(16, 32, 32, 16)),
        ((64, 0), 64, 16, QuantGroupedTiling(32, 32, 32, 8)),
        ((0, 0, 0), 32, 16, QuantGroupedTiling(16, 32, 32, 4)),
        ((289,), 608, 16, QuantGroupedTiling(304, 32, 32, 18)),
    ],
)
def test_rectangular_cover_and_work_conservation_on_diverse_geometry(
    rows, n, k, tiling
):
    schedule = plan_quant_grouped(rows, n, k, tiling)
    assert schedule.total_m == sum(rows)
    tiles = _all_tiles(schedule)
    assert sum(2 * tile.m * tile.n * tile.k for tile in tiles) == (
        2 * sum(rows) * n * k
    )

    prefix_rows = [0]
    for value in rows:
        prefix_rows.append(prefix_rows[-1] + value)
    for group, group_rows in enumerate(rows):
        if group_rows == 0:
            assert not any(tile.group == group for tile in tiles)
            continue
        group_tiles = [tile for tile in tiles if tile.group == group]
        cells = set()
        for tile in group_tiles:
            assert tile.m > 0 and tile.n > 0
            assert tile.k == k
            assert tile.n_offset % 32 == 0 and tile.n % 32 == 0
            assert tile.x_offset == (
                prefix_rows[group] * k + tile.m_offset * k
            )
            assert tile.weight_offset == group * n * k + tile.n_offset * k
            assert tile.output_offset == (
                prefix_rows[group] * n + tile.m_offset * n + tile.n_offset
            ) * 2
            assert tile.row_scale_offset == (prefix_rows[group] + tile.m_offset) * 4
            assert tile.column_scale_offset == (group * n + tile.n_offset) * 4
            for m in range(tile.m_offset, tile.m_offset + tile.m):
                for column in range(tile.n_offset, tile.n_offset + tile.n):
                    cells.add((m, column))
        assert len(cells) == group_rows * n
        assert sum(tile.m * tile.n for tile in group_tiles) == group_rows * n

    for owner in range(tiling.used_cores):
        owner_tiles = _tiles(schedule, owner)
        assert len(owner_tiles) == schedule.owner_tile_counts[owner]
        assert [tile.ordinal for tile in owner_tiles] == list(
            range(len(owner_tiles))
        )


def test_schedule_order_is_independent_of_declared_base_k():
    rows, n, k = (64, 0, 32), 64, 16
    first = plan_quant_grouped(rows, n, k, QuantGroupedTiling(32, 32, 32, 8))
    second = plan_quant_grouped(rows, n, k, QuantGroupedTiling(32, 32, 128, 8))
    assert [
        [astuple(tile) for tile in first.iter_owner_tiles(owner)]
        for owner in range(8)
    ] == [
        [astuple(tile) for tile in second.iter_owner_tiles(owner)]
        for owner in range(8)
    ]
    assert first.summary()["tiling"]["base_k"] == 32
    assert second.summary()["tiling"]["base_k"] == 128


def test_huge_tile_grid_is_consumed_lazily_without_bounded_allocation():
    huge_rows = 10**12
    tracemalloc.start()
    try:
        schedule = plan_quant_grouped(
            (huge_rows,), 32, 16, QuantGroupedTiling(16, 32, 32, 8)
        )
        first = next(schedule.iter_owner_tiles(0))
        rest = list(itertools.islice(schedule.iter_owner_tiles(0), 4))
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert first.ordinal == 0
    assert [tile.ordinal for tile in rest] == [0, 1, 2, 3]
    assert schedule.owner_tile_counts == (10**12 // 16 // 8,) * 8
    # A materialized grid here would contain >10^9 tile records.
    assert peak < 8 * 1024 * 1024


def test_rotated_start_final_split_is_rejected_instead_of_misordered():
    # Direct C++ execution of the pinned schedule yields duplicated N=128
    # rectangles on owners 1 and 3 for group 1.  Do not invent a repair.
    with pytest.raises(ValueError, match="unsupported_last_group_split_with_rotated_start"):
        plan_quant_grouped(
            (1, 1), 256, 16, QuantGroupedTiling(16, 256, 32, 8)
        )
    # The same rotation is well-defined when source N granularity rules out
    # an actual split.  A 32-element NZ block must not spuriously reject it.
    unsplit = plan_quant_grouped(
        (1, 1), 128, 16, QuantGroupedTiling(16, 128, 32, 8)
    )
    assert unsplit.owner_tile_counts == (1, 1, 0, 0, 0, 0, 0, 0)
