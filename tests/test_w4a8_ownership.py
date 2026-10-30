"""Independent owner-distribution checks for the A8W4 MSD source schedules."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, predict
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.w4a8_schedule import (
    down_schedule, gate_schedule, source_row_partition, strided_rows, total_work,
)


ROOT = Path(__file__).resolve().parents[1]
GATE_CASE = ROOT / "examples/source-families/w4a8-msd-control.json"
DOWN_CASE = ROOT / "examples/source-families/w4a8-down-msd-control.json"


def _gate_oracle(rows, n, k, aic):
    """Literal rolling preCount/curBlock loop from the mid kernel."""
    owner_tiles, owner_flops, owner_weight = {}, {}, {}
    pre_count = 0
    for value in rows:
        if value <= 0:
            continue
        block_m = -(-(2 * value) // 128)
        block_n = -(-n // 256)
        cur_count = pre_count + block_m * block_n
        for owner in range(aic):
            block = owner if owner >= pre_count else owner + aic
            while block < cur_count:
                local = block - pre_count
                m_index, n_index = divmod(local, block_n)
                m_tile = min(128, 2 * value - m_index * 128)
                n_tile = min(256, n - n_index * 256)
                owner_tiles[owner] = owner_tiles.get(owner, 0) + 1
                owner_flops[owner] = owner_flops.get(owner, 0) + 2 * m_tile * n_tile * k
                owner_weight[owner] = owner_weight.get(owner, 0) + 0.5 * n_tile * k
                block += aic
        pre_count = cur_count % aic
    return owner_tiles, owner_flops, owner_weight


def _down_oracle(rows, n, k, aic, base_m, base_n):
    owner_tiles, owner_flops, owner_weight = {}, {}, {}
    pre_count = 0
    for value in rows:
        if value <= 0:
            continue
        block_m = -(-(2 * value) // base_m)
        block_n = -(-n // base_n)
        cur_count = pre_count + block_m * block_n
        for owner in range(aic):
            block = owner if owner >= pre_count else owner + aic
            while block < cur_count:
                local = block - pre_count
                m_index, n_index = divmod(local, block_n)
                m_tile = min(base_m, 2 * value - m_index * base_m)
                n_tile = min(base_n, n - n_index * base_n)
                owner_tiles[owner] = owner_tiles.get(owner, 0) + 1
                owner_flops[owner] = owner_flops.get(owner, 0) + 2 * m_tile * n_tile * k
                owner_weight[owner] = owner_weight.get(owner, 0) + 0.5 * n_tile * k
                block += aic
        pre_count = cur_count % aic
    return owner_tiles, owner_flops, owner_weight


def _assist_oracle(rows, owners):
    """Naive per-owner row walk: one assist load per group encountered."""
    ranges, start = [], 0
    for value in rows:
        if value:
            ranges.append((start, start + value))
        start += value
    loads = {}
    for owner, start, count in source_row_partition(sum(rows), owners):
        loads[owner] = sum(1 for lo, hi in ranges
                           if lo < start + count and hi > start)
    return loads


def _real_gate_op():
    raw = json.loads(GATE_CASE.read_text())
    device = raw["machine"]["device"]
    device["compute_units"] = 20
    device["resource_units"] = {
        "cube": 20, "vector": 40, "mte2_aic": 20, "mte2_aiv": 40, "mte3": 40,
    }
    case = parse_case(raw)
    fused = next(op for op in target_operators(case, phase="prefill", tokens=4)
                 if op.recipe == "ascend_w4a8_msd")
    rows = [1] * 8 + [0] * 248
    metadata = dict(fused.metadata)
    config = dict(metadata["w4a8_msd"])
    config.update(
        m=8, n=512, k=6144, expert_rows=rows, active_groups=8,
        expert_scan_count=256, zero_expert_indices=list(range(8, 256)),
        zero_bank_offsets_int4_elements=[index * 512 * 6144 for index in range(8, 256)],
    )
    metadata["w4a8_msd"] = config
    return case, replace(fused, local_shape=(8, 512, 6144),
                         metadata=tuple(metadata.items()))


def test_gate_rolling_tile_oracle_and_e256_m8_versus_one_active():
    rows = (1,) * 8 + (0,) * 248
    schedule = gate_schedule(8, 512, 6144, rows, aic_owners=20, aiv_owners=40)
    tiles, flops, weight = _gate_oracle(rows, 512, 6144, 20)
    assert schedule.expert_scan_count == 256 and schedule.active_experts == 8
    assert schedule.tile_count == 16
    assert len(schedule.zero_bank_offsets) == 248
    assert schedule.zero_bank_offsets[0] == 8 * 512 * 6144
    assert {record.owner: record.tiles for record in schedule.matrix} == tiles
    for record in schedule.matrix:
        assert record.work.matrix["int4"] == pytest.approx(flops[record.owner])
        assert record.work.resources["hbm"] >= weight[record.owner]
    matrix_weight = sum(record.work.resources.get("mte2", 0) for record in schedule.matrix)
    assert matrix_weight == pytest.approx(
        8 * 0.5 * 512 * 6144 + 8 * 8 * 512 + 2 * 8 * 6144)
    assert matrix_weight < 256 * 0.5 * 512 * 6144 + 256 * 8 * 512 + 8 * 512

    one = gate_schedule(8, 512, 6144, (8,) + (0,) * 255,
                        aic_owners=20, aiv_owners=40)
    one_tiles, one_flops, _ = _gate_oracle((8,) + (0,) * 255, 512, 6144, 20)
    assert one.tile_count == 2
    assert {record.owner: record.tiles for record in one.matrix} == one_tiles
    for record in one.matrix:
        assert record.work.matrix["int4"] == pytest.approx(one_flops[record.owner])
    assert sum(record.work.resources.get("mte2", 0) for record in one.matrix) == pytest.approx(
        0.5 * 512 * 6144 + 8 * 512 + 2 * 8 * 6144)


def test_gate_middle_zero_n_and_m_tails_with_owner_segments():
    rows = (3, 0, 129, 2)
    m, n, k = sum(rows), 320, 256
    schedule = gate_schedule(m, n, k, rows, aic_owners=20, aiv_owners=40)
    tiles, flops, _ = _gate_oracle(rows, n, k, 20)
    assert {record.owner: record.tiles for record in schedule.matrix} == tiles
    for record in schedule.matrix:
        assert record.work.matrix["int4"] == pytest.approx(flops[record.owner])
    assert sum(record.work.matrix["int4"] for record in schedule.matrix) == pytest.approx(
        4 * m * n * k)
    assert schedule.zero_bank_offsets == (n * k,)
    tail = schedule.experts[2]
    assert tail.double_rows == 258 and tail.m_tiles == 3
    assert tail.last_m_tile == 2 and tail.last_n_tile == 64
    loads = _assist_oracle(rows, 40)
    assert {record.owner: record.assist_loads for record in schedule.post_dequant} == loads
    assert sum(loads.values()) > schedule.active_experts


def test_gate_row_partition_eight_of_forty_aivs_and_four_aiv_split():
    rows = (1,) * 8 + (0,) * 248
    wide = gate_schedule(8, 512, 6144, rows, aic_owners=20, aiv_owners=40)
    narrow = gate_schedule(8, 512, 6144, rows, aic_owners=2, aiv_owners=4)
    assert [(record.owner, record.rows) for record in wide.pre] == [
        (owner, 1) for owner in range(8)]
    assert [(record.owner, record.rows) for record in wide.post_dequant] == [
        (owner, 1) for owner in range(8)]
    assert [(record.owner, record.rows, record.assist_loads)
            for record in narrow.pre] == [
        (0, 2, 0), (1, 2, 0), (2, 2, 0), (3, 2, 0)]
    assert [(record.owner, record.rows) for record in narrow.post_dequant] == [
        (0, 2), (1, 2), (2, 2), (3, 2)]
    assert source_row_partition(8, 40) == tuple((owner, owner, 1) for owner in range(8))
    assert strided_rows(8, 40) == tuple((owner, 1) for owner in range(8))


def test_down_two_aiv_tile_split_assist_duplication_and_host_bases():
    rows = (1,) * 8 + (0,) * 248
    schedule = down_schedule(8, 6144, 256, rows, aic_owners=20, aiv_owners=40)
    assert (schedule.base_m, schedule.base_n) == (64, 512)
    assert schedule.aivs_per_aic == 2 and schedule.tile_count == 96
    tiles, flops, weight = _down_oracle(rows, 6144, 256, 20, 64, 512)
    assert {record.owner: record.tiles for record in schedule.matrix} == tiles
    for record in schedule.matrix:
        assert record.work.matrix["int4"] == pytest.approx(flops[record.owner])
    assist = {record.owner: record.assist_loads for record in schedule.post_dequant}
    active = {record.owner: record.tiles for record in schedule.post_dequant}
    for aic in range(20):
        pair = (2 * aic, 2 * aic + 1)
        assert assist[pair[0]] == assist[pair[1]] == tiles.get(aic, 0)
        assert active.get(pair[0], 0) > 0 and active.get(pair[1], 0) > 0
        assert sum(active.get(unit, 0) for unit in pair) == tiles.get(aic, 0)
    assert sum(assist.values()) == 2 * schedule.tile_count
    assert sum(active.values()) == schedule.tile_count
    assert sum(record.work.matrix["int4"] for record in schedule.matrix) == pytest.approx(
        4 * 8 * 6144 * 256)
    assert sum(record.work.resources.get("mte2", 0) for record in schedule.matrix) == pytest.approx(
        8 * 0.5 * 6144 * 256 + 8 * 8 * 6144 + 12 * 8 * 256)
    assert total_work(schedule.post_dequant).resources["vector"] == pytest.approx(
        3 * 16 * 6144)

    tall = down_schedule(256, 640, 256, (33, 0, 97, 126),
                         aic_owners=20, aiv_owners=40)
    assert (tall.base_m, tall.base_n) == (128, 256)
    assert tall.experts[2].last_m_tile == 66
    assert tall.experts[0].last_n_tile == 128
    tall_tiles, _, _ = _down_oracle((33, 0, 97, 126), 640, 256, 20, 128, 256)
    assert {record.owner: record.tiles for record in tall.matrix} == tall_tiles

    single = down_schedule(8, 64, 64, (3, 0, 3, 2), aic_owners=4, aiv_owners=8)
    assert single.aivs_per_aic == 1
    assert total_work(single.post_dequant).resources["hbm"] == pytest.approx(
        4 * 8 * 64 + 4 * 64 * 3 + 4 * 8)


def test_public_real_m8_geometry_uses_schedule_work_and_stays_live():
    case, op = _real_gate_op()
    external = {dep: 0 for dep in op.deps}
    report = graph_costs(case, [op], external_inputs=external,
                         include_submission=True, isolated=True)
    schedule = gate_schedule(8, 512, 6144, (1,) * 8 + (0,) * 248,
                             aic_owners=20, aiv_owners=40)
    expected = total_work(schedule.pre, schedule.matrix, schedule.post_dequant,
                          schedule.swiglu, schedule.quantize)
    applied = report["isolated"]["operators"][0]["operator"]
    assert report["execution"]["recipes"] == {"ascend_w4a8_msd": 1}
    assert applied["flops"] == pytest.approx(expected.matrix["int4"])
    assert applied["hbm_bytes"] == pytest.approx(expected.resources["hbm"])
    assert applied["mte3_bytes"] == pytest.approx(expected.resources["mte3"])
    assert applied["vector_ops"] == pytest.approx(expected.resources["vector"])
    assert report["execution"]["serviced"]["matrix@r0:s0"] == pytest.approx(
        1000 * expected.matrix["int4"] / 500_000_000.0)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert report["timeline"][0]["ready_ms"] == pytest.approx(0)

    late = graph_costs(case, [op], external_inputs={dep: 5 for dep in op.deps},
                       include_submission=True, isolated=True)
    assert late["timeline"][0]["ready_ms"] == pytest.approx(5)
    assert late["timeline"][0]["complete_ms"] - report["timeline"][0]["complete_ms"] \
        == pytest.approx(5)

    twin = replace(op, op_id=op.op_id + ".twin")
    shared = graph_costs(case, [op, twin], external_inputs=external,
                         include_submission=True, isolated=True)
    assert shared["execution"]["work"]["flops"] == pytest.approx(
        2 * expected.matrix["int4"])
    assert shared["execution"]["work"]["hbm_bytes"] == pytest.approx(
        2 * expected.resources["hbm"])
    assert shared["execution"]["serviced"]["matrix@r0:s0"] == pytest.approx(
        2 * 1000 * expected.matrix["int4"] / 500_000_000.0)
    assert shared["execution"]["live_tasks"] == shared["execution"]["live_programs"] == 0

    raw = json.loads(GATE_CASE.read_text())
    predictions = predict(raw)
    assert predictions["execution"]["recipes"]["ascend_w4a8_msd"] == 3


def test_public_small_example_preserves_selected_recipes_and_owner_work():
    raw = json.loads(DOWN_CASE.read_text())
    report = predict(raw)
    assert report["execution"]["recipes"]["ascend_w4a8_msd"] == 3
    assert report["execution"]["recipes"]["ascend_w4a8_msd_grouped_down"] == 3
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


def _source_down_output_oracle(rows, n, k, cores):
    """Execute the source's per-core loops, including persistent vecCount."""
    avg_m = sum(rows) // len(rows)
    base_m, base_n = (64, 512) if avg_m < 32 else (128, 256)
    ratio = 1 if avg_m <= 128 and len(rows) <= 4 and k <= 8192 and n <= 8192 else 2
    output_elements = {}
    for core in range(cores):
        pre_count, vec_count = 0, 0
        for count in rows:
            if not count:
                continue
            block_n = (n + base_n - 1) // base_n
            block_m = (2 * count + base_m - 1) // base_m
            cur_count = pre_count + block_n * block_m
            cur_block = core if core >= pre_count else core + cores
            while cur_block < cur_count:
                m_idx, n_idx = divmod(cur_block - pre_count, block_n)
                cube_m = min(base_m // 2, count - m_idx * (base_m // 2))
                cube_n = min(base_n, n - n_idx * base_n)
                step = min(8192 // base_n, cube_m)
                for offset in range(0, cube_m, step):
                    vec_count += 1
                    sub = vec_count % ratio
                    physical = 2 * core + sub
                    elements = min(step, cube_m - offset) * cube_n
                    output_elements[physical] = output_elements.get(physical, 0) + elements
                cur_block += cores
            pre_count = cur_count % cores
    return output_elements


@pytest.mark.parametrize("rows,n,k,cores", [
    ((33,) + (0,) * 255, 512, 256, 1),
    ((17, 0, 65, 3) + (0,) * 5, 576, 256, 3),
    ((129, 0, 2, 3), 640, 256, 2),
    ((1,) * 8 + (0,) * 248, 6144, 256, 20),
])
def test_down_ub_chunk_owner_matches_independent_source_loop(rows, n, k, cores):
    schedule = down_schedule(sum(rows), n, k, rows,
                             aic_owners=cores, aiv_owners=2 * cores)
    expected = _source_down_output_oracle(rows, n, k, cores)
    assert {r.owner: r.work.resources["mte3"] / 2 for r in schedule.output} == expected
    assert sum(expected.values()) == sum(rows) * n
    if sum(rows) == 33:
        # Two full 16-row UB chunks then one tail alternate 1,0,1.
        # Whole-tile alternation would incorrectly assign 32/1 rows.
        assert expected == {0: 16 * n, 1: 17 * n}


def test_down_metadata_precedes_row_guard_and_keeps_idle_aivs():
    schedule = down_schedule(8, 6144, 256, (1,) * 8 + (0,) * 248,
                             aic_owners=20, aiv_owners=40)
    assert len(schedule.pre) == 40
    assert sum(r.rows > 0 for r in schedule.pre) == 8
    for record in schedule.pre[8:]:
        assert record.rows == 0
        assert record.work.resources["hbm"] == record.work.resources["mte2"] == 256 * 8
        assert record.work.resources["vector"] == 256 + 255 + 1
    pre = total_work(schedule.pre).resources
    assert pre["hbm"] == 2 * 8 * 256 + 40 * 256 * 8
    assert pre["vector"] == 6.5 * 8 * 256 + 40 * (256 + 255 + 1)


def test_public_source_selection_rejects_incompatible_vector_inventory():
    raw = json.loads(GATE_CASE.read_text())
    raw["machine"]["device"]["resource_units"]["vector"] = 4
    case = parse_case(raw)
    op = next(op for op in target_operators(case, phase="prefill", tokens=4)
              if op.recipe == "ascend_w4a8_msd")
    with pytest.raises(ValueError, match="two_physical_aivs_per_aic"):
        graph_costs(case, [op], external_inputs={dep: 0 for dep in op.deps})
