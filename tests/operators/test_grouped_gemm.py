"""Independent histogram, tile, byte, address and small-matrix oracles.

The planner is source-derived, not capture-fitted.  These checks establish the
selected branch, semantic histogram compaction, literal raw-list termination,
per-expert skew/tails, static core assignment, per-unit buffer ownership,
physical-stage placement and the public execution work ledger.  They are not an
accuracy claim against a device measurement.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.operators.grouped_gemm import (
    build,
    compact_histogram,
    host_tiling,
    interpret_raw_group_list,
    iter_tiles,
    make_op,
    pad_raw_group_list,
)
from mindie_sim.prediction.program import ParallelProgram, RecipeContext
from mindie_sim.prediction.verification import verify_operator
from tools.run_operator_cases import control_case, run

ROOT = Path(__file__).resolve().parents[2]
CASE_FILE = ROOT / "tests/fixtures/operator_cases/grouped_gemm.json"
MANIFEST_FILE = ROOT / "tests/fixtures/operator_sources/grouped_gemm.json"

SINGLE_M = 128
SINGLE_N = 256
ELEM = 2.0

FULL_BUFFERS = {"l1": 524288, "l0a": 65536, "l0b": 65536, "l0c": 262144,
                "ub": 196608}


def cases():
    return json.loads(CASE_FILE.read_text())["cases"]


def params_for(rows, n, k, **extra):
    values = {
        "expert_rows": list(rows),
        "n": n,
        "k": k,
        "dtype": "bf16",
        "implementation": "ascendc_grouped_nd_bf16",
        "transpose_weight": False,
        "weight_layout": "ND",
        "tiling_key": 10,
    }
    values.update(extra)
    return values


def hardware(**overrides):
    units = overrides.pop("units", 32)
    split_copy = overrides.pop("split_copy", False)
    buffers = overrides.pop("buffers", None)
    buffer_units = overrides.pop("buffer_units", None)
    case = control_case(units=units, split_copy=split_copy)
    device = case["machine"]["device"]
    if buffers is not None:
        device["buffers"] = buffers
    if buffer_units is not None:
        device["buffer_units"] = buffer_units
    device.update(overrides)
    return case


def reference_partition(core, owners, units):
    """Independent full-width physical partition oracle.

    ``owners`` is the physical compute width, not the launched/active block
    count.  A resource (copy engine or pool bank) is either split evenly among
    fixed owner groups, shared by an integer number of adjacent owners, or the
    declared geometry is incompatible.
    """
    if units >= owners and units % owners == 0:
        width = units // owners
        return tuple(range(core * width, (core + 1) * width))
    if owners % units == 0:
        return (core // (owners // units),)
    raise ValueError(f"incompatible_physical_partition:{owners}:{units}")


def copy_bound_hardware(units, copy_units, *, per_unit=1e9):
    """Synthetic hardware where only declared copy-engine service is binding."""
    case = hardware(units=units, split_copy=True)
    device = case["machine"]["device"]
    device["rates"] = {"memory": 1e20, "vector": 1e20,
                       "read": copy_units * per_unit, "write": 1e20}
    device["compute_peaks"] = dict.fromkeys(device["compute_peaks"], 1e20)
    device["resource_units"] = {"vector": 2 * units, "mte2": copy_units,
                                "mte3": copy_units}
    return case


def placement_context(op, cores, copy_units, buffers=None, buffer_units=None):
    buffers = buffers or {}
    return RecipeContext(
        op, {},
        {"cube": "matrix", "hbm": "memory", "mte2": "read", "mte3": "write"},
        {"matrix": 1.0, "memory": 1e20, "read": 1e20, "write": 1e20}, cores,
        pool_names={pool: f"{pool}@r0:s0" for pool in buffers},
        resource_units={"cube": cores, "mte2": copy_units, "mte3": copy_units},
        compute_peaks={"bf16": 1e20},
        pool_units=dict(buffer_units or {}),
    )


def reference_entries(rows, group_ids, n, k, weight_experts=None):
    """Independent source-order histogram interpreter with per-entry tiling."""
    n_tiles = math.ceil(n / SINGLE_N)
    entries, prefix = [], 0
    for histogram_index, (expert_id, count) in enumerate(zip(group_ids, rows)):
        if count == 0:
            continue
        m_tiles = math.ceil(count / SINGLE_M)
        a = count * k * ELEM * n_tiles
        b = m_tiles * k * n * ELEM
        c = count * n * ELEM
        entries.append({
            "index": len(entries),
            "histogram_index": histogram_index,
            "expert_id": expert_id,
            "rows": count,
            "x_row_offset": prefix,
            "y_row_offset": prefix,
            "x_element_offset": prefix * k,
            "y_element_offset": prefix * n,
            "w_element_offset": expert_id * k * n,
            "m_tiles": m_tiles,
            "n_tiles": n_tiles,
            "tasks": m_tiles * n_tiles,
            "flops": 2 * count * n * k,
            "a_bytes": a,
            "b_bytes": b,
            "c_bytes": c,
            "mte2_bytes": a + b,
            "mte3_bytes": c,
            "hbm_bytes": a + b + c,
        })
        prefix += count
    return entries, prefix


def reference_tiles(rows, group_ids, n, k):
    """Enumerate the kernel's flat mIdx*blockDimN+nIdx task order independently."""
    n_tiles = math.ceil(n / SINGLE_N)
    records, prefix, group_index = [], 0, 0
    for histogram_index, (expert_id, count) in enumerate(zip(group_ids, rows)):
        if count == 0:
            continue
        m_tiles = math.ceil(count / SINGLE_M)
        for m_index in range(m_tiles):
            tile_rows = min(SINGLE_M, count - m_index * SINGLE_M)
            for n_index in range(n_tiles):
                tile_cols = min(SINGLE_N, n - n_index * SINGLE_N)
                records.append({
                    "group_index": group_index,
                    "histogram_index": histogram_index,
                    "expert_id": expert_id,
                    "task_index": m_index * n_tiles + n_index,
                    "m_index": m_index,
                    "n_index": n_index,
                    "tile_rows": tile_rows,
                    "tile_cols": tile_cols,
                    "x_row_offset": prefix + m_index * SINGLE_M,
                    "y_row_offset": prefix + m_index * SINGLE_M,
                    "x_element_offset": (prefix + m_index * SINGLE_M) * k,
                    "y_element_offset": (prefix + m_index * SINGLE_M) * n + n_index * SINGLE_N,
                    "w_element_offset": expert_id * k * n + n_index * SINGLE_N,
                    "weight_row_stride": n,
                })
        prefix += count
        group_index += 1
    return records


def reference_assignment(rows, n, core_num):
    """Independent rotation: task t of a compacted group lands on (t+count)%cores."""
    n_tiles = math.ceil(n / SINGLE_N)
    assigned, cumulative, group_index = [], 0, 0
    for count in rows:
        if count == 0:
            continue
        tasks = math.ceil(count / SINGLE_M) * n_tiles
        start = cumulative % core_num
        for task in range(tasks):
            assigned.append((group_index, task, (task + start) % core_num))
        cumulative += tasks
        group_index += 1
    return assigned


def test_development_and_heldout_cases_select_only_the_declared_bf16_nd_key10_branch():
    selected = 0
    splits = []
    for case in cases():
        op = make_op(case["params"])
        dispatch = dict(op.metadata)["dispatch"]
        assert {key: dispatch[key] for key in case["expected_dispatch"]} == case["expected_dispatch"]
        assert dispatch["implementation"] == "ascendc_grouped_nd_bf16"
        assert dispatch["transpose_weight"] is False
        assert dispatch["weight_layout"] == "ND"
        assert dispatch["tiling_key"] == 10
        assert dispatch["quantization"] == "none"
        assert dispatch["group_list_mode"] == "key_value_2d"
        assert dispatch["input_adapter"] == (
            "semantic_histogram_compacted_preserve_expert_ids")
        assert dispatch["raw_group_list_terminator"] == (
            "first_row_count_nonpositive_breaks_remaining_list")
        assert op.dtype == "bf16" and op.layout == "ND"
        selected += 1
        splits.append(case["split"])
    assert selected == len(cases()) == 5
    assert splits == ["development"] * 3 + ["heldout_geometry"] * 2


def test_unselected_branches_and_invalid_histogram_inputs_are_rejected_explicitly():
    base = params_for([1, 0, 3], 128, 64)
    for key, value in (
        ("implementation", "some_other_grouped_gemm"),
        ("weight_layout", "NZ"),
        ("weight_format", "NZ"),
        ("transpose_weight", True),
        ("tiling_key", 11),
        ("dtype", "fp16"),
        ("quantization", "w8a8"),
    ):
        with pytest.raises(ValueError):
            make_op({**base, key: value})
    with pytest.raises(ValueError, match="total_rows_zero_invalid_host_tiling"):
        make_op(params_for([0, 0, 0], 128, 64))
    with pytest.raises(ValueError, match="group_ids_must_address_declared_weight_experts"):
        make_op(params_for([1, 2], 8, 4, group_ids=[0, 2]))
    with pytest.raises(ValueError, match="group_ids_must_address_declared_weight_experts"):
        make_op(params_for([1, 2], 8, 4, group_ids=[0, 2], weight_experts=2))
    with pytest.raises(ValueError, match="group_ids_length_must_match_expert_rows"):
        make_op(params_for([1, 2], 8, 4, group_ids=[0]))
    with pytest.raises(ValueError, match="group_ids_must_be_unique_histogram_bins"):
        make_op(params_for([1, 2], 8, 4, group_ids=[0, 0]))
    with pytest.raises(ValueError):
        make_op({**base, "n": 0})
    with pytest.raises(ValueError):
        make_op({**base, "k": 0})
    with pytest.raises(ValueError):
        make_op({**base, "expert_rows": [1, -1, 3]})
    with pytest.raises(ValueError, match="expert_rows_must_not_be_empty"):
        make_op({**base, "expert_rows": []})
    with pytest.raises(ValueError, match="x_rows_must_equal_histogram_sum"):
        make_op({**base, "x_rows": 3})
    with pytest.raises(ValueError, match="group_ids_must_address_declared_weight_experts"):
        make_op(params_for([1, 2], 8, 4, group_ids=[0, 5], weight_experts=4))


def test_histogram_compaction_processes_all_nonempty_bins_without_missing_rows():
    expected = {
        "grouped_gemm-0": (4, [0, 2], [1, 3]),
        "grouped_gemm-1": (12, [0, 1, 2], [4, 4, 4]),
        "grouped_gemm-2": (65, [0, 1], [64, 1]),
        "grouped_gemm-3": (84, [0, 2, 3], [17, 2, 65]),
        "grouped_gemm-4": (136, [0, 2], [129, 7]),
    }
    for case in cases():
        op = make_op(case["params"])
        meta = dict(op.metadata)
        input_rows, ids, counts = expected[case["id"]]
        assert meta["input_rows"] == input_rows == sum(case["params"]["expert_rows"])
        assert meta["processed_m"] == input_rows
        assert meta["missing_rows"] == 0
        assert meta["host"]["processed_m"] == input_rows
        assert meta["host"]["total_m"] == input_rows
        assert [entry["expert_id"] for entry in meta["processed_entries"]] == ids
        assert [entry["rows"] for entry in meta["processed_entries"]] == counts
        assert meta["unprocessed_entries"] == ()
        assert op.semantic_shape[0] == op.local_shape[0] == op.launch_shape[0] == input_rows
        assert meta["raw_group_list"] == meta["kernel_group_list"]
        assert meta["raw_group_list_active"] == meta["raw_group_list_active"]
        assert meta["raw_group_list_extent"] == len(case["params"]["expert_rows"])
        assert meta["raw_group_list_extent"] == len(case["params"]["expert_rows"])
        assert len(meta["raw_group_list"]) == meta["weight_experts"]
        if len(ids) < meta["weight_experts"]:
            assert meta["raw_group_list"][len(ids):] == ((0, 0),) * (
                meta["weight_experts"] - len(ids))
            assert meta["stop_reason"] == "split_value_nonpositive"
            assert meta["stop_index"] == len(ids)
        else:
            assert meta["stop_reason"] == "end_of_list"
            assert meta["stop_index"] == meta["weight_experts"]


def test_review_counterexamples_now_process_4_84_and_136_rows():
    cases_rows = (
        ([1, 0, 3], 128, 64, 4, [(0, 1), (2, 3)]),
        ([17, 0, 2, 65], 256, 64, 84, [(0, 17), (2, 2), (3, 65)]),
        ([129, 0, 7], 128, 128, 136, [(0, 129), (2, 7)]),
    )
    for rows, n, k, total, active in cases_rows:
        op = make_op(params_for(rows, n, k))
        meta = dict(op.metadata)
        plan = build(op, Geometry(32, {}, {}))
        assert meta["input_rows"] == meta["processed_m"] == total
        assert meta["missing_rows"] == 0
        assert meta["histogram_positive_entries"] == tuple(active)
        assert plan.work.matrix["bf16"] == 2 * total * n * k
        assert plan.work.matrix["bf16"] != 2 * sum(rows[:1]) * n * k


def test_leading_and_trailing_empty_bins_do_not_drop_positive_rows():
    leading = make_op(params_for([0, 5, 7], 16, 8))
    leading_meta = dict(leading.metadata)
    assert leading_meta["input_rows"] == leading_meta["processed_m"] == 12
    assert leading_meta["histogram_positive_entries"] == ((1, 5), (2, 7))
    assert [(entry["expert_id"], entry["x_row_offset"])
            for entry in leading_meta["processed_entries"]] == [(1, 0), (2, 5)]
    assert build(leading, Geometry(32, {}, {})).work.matrix["bf16"] == 2 * 12 * 16 * 8

    trailing = make_op(params_for([5, 7, 0], 16, 8))
    trailing_meta = dict(trailing.metadata)
    assert trailing_meta["input_rows"] == trailing_meta["processed_m"] == 12
    assert trailing_meta["histogram_positive_entries"] == ((0, 5), (1, 7))
    assert trailing_meta["kernel_group_list"] == ((0, 5), (1, 7), (0, 0))
    assert trailing_meta["stop_index"] == 2
    assert build(trailing, Geometry(32, {}, {})).work.matrix["bf16"] == 2 * 12 * 16 * 8


def test_literal_raw_group_list_terminates_but_histogram_zero_does_not():
    # The kernel's literal raw list stops at the first count <= 0.
    raw = ((0, 2), (1, 0), (2, 3))
    interpreted = interpret_raw_group_list(raw, n=4, k=3)
    assert interpreted["processed_rows"] == 2
    assert [entry["expert_id"] for entry in interpreted["processed"]] == [0]
    assert interpreted["stop_index"] == 1
    assert interpreted["stop_reason"] == "split_value_nonpositive"
    assert [entry["expert_id"] for entry in interpreted["unprocessed"]] == [1, 2]
    assert interpreted["counts_are_direct_not_cumulative"] is True

    # Counts are direct row counts, not cumulative end offsets: 2 then 3 rows.
    direct = interpret_raw_group_list(((0, 2), (1, 3)), n=5, k=2)
    assert direct["processed_rows"] == 5
    assert [entry["x_row_offset"] for entry in direct["processed"]] == [0, 2]
    assert direct["stop_reason"] == "end_of_list"
    # A cumulative reading would have consumed only rows 0..2, not five rows.
    assert direct["processed_rows"] != 3

    negative = interpret_raw_group_list(((0, 5), (1, -1), (2, 7)))
    assert negative["processed_rows"] == 5
    assert negative["stop_index"] == 1

    # The semantic adapter deliberately differs: the zero bin is skipped.
    assert compact_histogram((1, 0, 3)) == ((0, 1), (2, 3))
    assert sum(count for _, count in compact_histogram((1, 0, 3))) == 4
    assert interpret_raw_group_list(((0, 1), (1, 0), (2, 3)))["processed_rows"] == 1
    assert pad_raw_group_list(compact_histogram((1, 0, 3)), 3) == (
        (0, 1), (2, 3), (0, 0))
    assert pad_raw_group_list(compact_histogram((1, 0, 3)), 5) == (
        (0, 1), (2, 3), (0, 0), (0, 0), (0, 0))


def test_host_tiling_matches_source_formulas_and_declares_per_unit_panels():
    expected = {
        4: (16, 64, 64, 4),
        12: (16, 64, 64, 4),
        65: (80, 64, 12, 4),
        84: (96, 64, 8, 4),
        136: (144, 64, 4, 4),
    }
    for total_m, (base_m, base_k, step_ka, step_kb) in expected.items():
        op = make_op(params_for([total_m], 128, 64))
        tiling = host_tiling(op, Geometry(32, {}, FULL_BUFFERS))
        assert tiling["base_n"] == 256
        assert tiling["single_m"] == 128 and tiling["single_n"] == 256
        assert tiling["base_m"] == base_m
        assert tiling["base_k"] == base_k
        assert tiling["step_ka"] == step_ka
        assert tiling["step_kb"] == step_kb
        assert tiling["depth_a1"] == 2 * step_ka
        assert tiling["depth_b1"] == 2 * step_kb
        assert tiling["status"] == "complete"
        assert tiling["core_num"] == min(32, total_m)
        allocations = tiling["allocation_bytes"]
        assert set(allocations) == {"l1", "l0a", "l0b", "l0c"}
        assert allocations["l1"] == (
            step_ka * base_m * base_k * ELEM + step_kb * 256 * base_k * ELEM)
        assert allocations["l0a"] == base_m * base_k * ELEM
        assert allocations["l0b"] == 256 * base_k * ELEM
        assert allocations["l0c"] == base_m * 256 * 4
        assert allocations["l1"] <= FULL_BUFFERS["l1"]
        assert allocations["l0a"] <= FULL_BUFFERS["l0a"]
        assert allocations["l0b"] <= FULL_BUFFERS["l0b"]
        assert allocations["l0c"] <= FULL_BUFFERS["l0c"]


def test_host_tiling_leaves_platform_dependent_fields_unresolved_without_buffers():
    op = make_op(params_for([1, 0, 3], 128, 64))
    tiling = host_tiling(op, Geometry(32, {}, {}))
    assert tiling["core_num"] == 4
    assert tiling["base_m"] is None and tiling["base_k"] is None
    assert tiling["step_ka"] is None and tiling["status"] == "unresolved_l0_buffers"
    assert tiling["allocation_bytes"] == {}

    l0_only = host_tiling(op, Geometry(32, {}, {
        "l0a": 65536, "l0b": 65536, "l0c": 262144}))
    assert l0_only["base_m"] == 16 and l0_only["base_k"] == 64
    assert l0_only["step_ka"] is None and l0_only["status"] == "l0_derived"
    assert set(l0_only["allocation_bytes"]) == {"l0a", "l0b", "l0c"}

    l1_only = host_tiling(op, Geometry(32, {}, {"l1": 524288, "ub": 196608}))
    assert l1_only["status"] == "unresolved_l0_buffers"
    assert l1_only["allocation_bytes"] == {}

    with pytest.raises(ValueError, match="host_tiling_l1_below_256k"):
        host_tiling(op, Geometry(32, {}, {
            "l1": 200000, "l0a": 65536, "l0b": 65536, "l0c": 262144}))
    with pytest.raises(ValueError, match="host_tiling_base_k_zero"):
        host_tiling(op, Geometry(32, {}, {
            "l1": 524288, "l0a": 65536, "l0b": 1024, "l0c": 262144}))


def test_per_expert_work_matches_independent_tile_and_byte_formulas():
    for case in cases():
        op = make_op(case["params"])
        params = case["params"]
        meta = dict(op.metadata)
        expected, prefix = reference_entries(
            meta["expert_rows"], meta["group_ids"], params["n"], params["k"],
            meta["weight_experts"])
        assert meta["processed_entries"] == tuple(expected)
        assert meta["host"]["processed_m"] == prefix == meta["input_rows"]
        plan = build(op, Geometry(32, {}, {}))
        assert plan.work.matrix == {"bf16": float(sum(row["flops"] for row in expected))}
        assert plan.work.resources == {
            "hbm": float(sum(row["hbm_bytes"] for row in expected)),
            "mte2": float(sum(row["mte2_bytes"] for row in expected)),
            "mte3": float(sum(row["mte3_bytes"] for row in expected)),
        }
        tiles = list(iter_tiles(op))
        assert sum(tile["flops"] for tile in tiles) == plan.work.matrix["bf16"]
        assert sum(tile["a_bytes"] + tile["b_bytes"] for tile in tiles) == plan.work.resources["mte2"]
        assert sum(tile["c_bytes"] for tile in tiles) == plan.work.resources["mte3"]
        assert sum(tile["a_bytes"] + tile["b_bytes"] + tile["c_bytes"]
                   for tile in tiles) == plan.work.resources["hbm"]
        assert meta["host"]["kernel_task_num"] == sum(
            entry["tasks"] for entry in expected)


def test_skew_and_tails_are_kept_per_tile_and_not_replaced_by_a_mean():
    rows, n, k = [129, 0, 7], 128, 128
    op = make_op(params_for(rows, n, k))
    tiles = list(iter_tiles(op))
    assert [tile["tile_rows"] for tile in tiles] == [128, 1, 7]
    assert [tile["tile_cols"] for tile in tiles] == [n, n, n]
    assert [tile["x_row_offset"] for tile in tiles] == [0, 128, 129]
    assert [tile["expert_id"] for tile in tiles] == [0, 0, 2]
    assert [tile["flops"] for tile in tiles] == [
        2 * 128 * n * k, 2 * 1 * n * k, 2 * 7 * n * k]
    plan = build(op, Geometry(32, {}, {}))
    assert plan.work.matrix["bf16"] == 2 * 136 * n * k
    assert plan.work.matrix["bf16"] != 2 * ((129 + 7) / 2) * n * k

    # Compute-dominant control: only the 128-row tile determines completion.
    fast_memory = hardware(compute_peaks={"bf16": 1e10, "fp32": 1e10},
                           rates={"memory": 1e18, "vector": 1e18})
    report = verify_operator(fast_memory, op, build)
    assert report["status"] == "passed"
    assert report["result"]["duration_ms"] == pytest.approx(
        32 * 1000 * 2 * 128 * n * k / 1e10, rel=1e-9)
    assert report["result"]["duration_ms"] != pytest.approx(
        32 * 1000 * 2 * 68 * n * k / 1e10, rel=1e-3)

    # N tails and three different M counts stay per-entry in task order.
    skew = make_op(params_for([17, 0, 2, 65], 300, 64))
    expected = reference_tiles([17, 0, 2, 65], [0, 1, 2, 3], 300, 64)
    actual = list(iter_tiles(skew))
    assert len(actual) == len(expected) == 6
    for got, want in zip(actual, expected):
        for key in ("group_index", "histogram_index", "expert_id", "task_index",
                    "m_index", "n_index", "tile_rows", "tile_cols",
                    "x_row_offset", "y_row_offset", "x_element_offset",
                    "y_element_offset", "w_element_offset", "weight_row_stride"):
            assert got[key] == want[key]
    assert [tile["tile_rows"] for tile in actual] == [17, 17, 2, 2, 65, 65]
    assert [tile["tile_cols"] for tile in actual] == [256, 44, 256, 44, 256, 44]


def test_multiple_tasks_on_one_core_serialize_at_the_physical_core_rate():
    rows, n, k = [4, 4, 4], 128, 128
    op = make_op(params_for(rows, n, k))
    assert host_tiling(op, Geometry(2, {}, {}))["core_num"] == 2
    assert [(tile["group_index"], tile["core"]) for tile in iter_tiles(op, 2)] == [
        (0, 0), (1, 1), (2, 0)]
    fast = hardware(units=2, compute_peaks={"bf16": 1e10, "fp32": 1e10},
                    rates={"memory": 1e18, "vector": 1e18})
    report = verify_operator(fast, op, build)
    assert report["status"] == "passed"
    per_task_ms = 2 * 1000 * 2 * 4 * n * k / 1e10
    assert report["result"]["duration_ms"] == pytest.approx(2 * per_task_ms, rel=1e-8)


def test_static_rotation_matches_the_kernel_flat_task_order_and_tails():
    rows, n, k = [130, 0, 2], 300, 5
    op = make_op(params_for(rows, n, k))
    expected = reference_tiles(rows, [0, 1, 2], n, k)
    actual = list(iter_tiles(op))
    assert len(actual) == len(expected) == 6
    for got, want in zip(actual, expected):
        for key in ("group_index", "histogram_index", "expert_id", "task_index",
                    "m_index", "n_index", "tile_rows", "tile_cols",
                    "x_row_offset", "y_row_offset", "x_element_offset",
                    "y_element_offset", "w_element_offset", "weight_row_stride"):
            assert got[key] == want[key]
    assert [tile["tile_rows"] for tile in actual] == [128, 128, 2, 2, 2, 2]
    assert [tile["tile_cols"] for tile in actual] == [256, 44, 256, 44, 256, 44]
    for core_num in (1, 2, 3, 4, 5, 32):
        assigned = [(tile["group_index"], tile["task_index"], tile["core"])
                    for tile in iter_tiles(op, core_num)]
        assert assigned == reference_assignment(rows, n, core_num)
    with pytest.raises(ValueError, match="core_num"):
        list(iter_tiles(op, 0))


def test_copy_ownership_uses_full_physical_width_not_the_launched_grid():
    # Review counterexample: 32 physical compute units, 8 launched CTAs.
    rows, n, k = [1] * 8, 128, 64
    op = make_op(params_for(rows, n, k))
    task_bytes = 1 * k * 2 + k * n * 2
    assert task_bytes == 16512
    expected_ms = {16: 0.033024, 32: 0.016512, 64: 0.008256}

    for copies in (16, 32, 64):
        program = build(op, Geometry(32, {}, {})).program(
            placement_context(op, 32, copies))
        assert len(program.programs) == 8
        per_unit_bytes = {}
        for core, child in enumerate(program.programs):
            load, compute, store = child.tile(0)
            expected_units = reference_partition(core, 32, copies)
            assert load.placement == {"read": expected_units}
            assert compute.placement == {"matrix": (core,)}
            assert store.placement == {"write": expected_units}
            for unit in expected_units:
                per_unit_bytes[unit] = (per_unit_bytes.get(unit, 0)
                                        + task_bytes / len(expected_units))

        # Independent copy-bound oracle: each declared 1 GB/s unit serves its
        # assigned share; the slowest unit bounds the whole kernel.
        oracle_ms = max(per_unit_bytes.values()) / 1e9 * 1000
        assert oracle_ms == pytest.approx(expected_ms[copies], rel=1e-12)
        report = verify_operator(copy_bound_hardware(32, copies), op, build)
        assert report["status"] == "passed"
        assert report["work_mismatches"] == {}
        assert report["result"]["duration_ms"] == pytest.approx(
            expected_ms[copies], rel=1e-6)
        if copies == 16:
            # The old launched-width candidate produced one unit per CTA and
            # 0.016512 ms; a shared unit must serialize the adjacent pair.
            assert report["result"]["duration_ms"] >= 0.033024
            assert report["result"]["duration_ms"] == pytest.approx(
                2 * expected_ms[32], rel=1e-6)


def test_one_and_32_compute_units_cover_physical_ownership_and_32_cta_scaling():
    rows, n, k = [1] * 32, 128, 64
    op = make_op(params_for(rows, n, k))
    task_bytes = 1 * k * 2 + k * n * 2

    # One real source core owns all 32 declared copy units.
    one_core = build(op, Geometry(1, {})).program(
        placement_context(op, 1, 32))
    assert len(one_core.programs) == 1
    load = one_core.programs[0].tile(0)[0]
    assert load.placement == {"read": tuple(range(32))}
    one_core_report = verify_operator(copy_bound_hardware(1, 32), op, build)
    assert one_core_report["status"] == "passed"
    # 32 serial tasks each use all 32 units: total bytes / total rate.
    assert one_core_report["result"]["duration_ms"] == pytest.approx(
        32 * task_bytes / (32 * 1e9) * 1000, rel=1e-9)

    # Actual 32-CTA launch: 32 source core ids, one unit each, distinct units.
    many = build(op, Geometry(32, {})).program(
        placement_context(op, 32, 32))
    assert len(many.programs) == 32
    placements = [child.tile(0)[0].placement["read"]
                  for child in many.programs]
    assert placements == [(core,) for core in range(32)]
    assert [child.tile(0)[1].placement["matrix"]
            for child in many.programs] == [(core,) for core in range(32)]
    assert len({unit for units in placements for unit in units}) == 32
    many_report = verify_operator(copy_bound_hardware(32, 32), op, build)
    assert many_report["status"] == "passed"
    assert many_report["result"]["duration_ms"] == pytest.approx(
        task_bytes / 1e9 * 1000, rel=1e-6)

    # Same per-unit physical rate, one CTA/one unit: all 32 loads serialize.
    one_cta = verify_operator(copy_bound_hardware(1, 1), op, build)
    assert one_cta["status"] == "passed"
    assert one_cta["result"]["duration_ms"] == pytest.approx(
        32 * task_bytes / 1e9 * 1000, rel=1e-6)
    assert one_cta["result"]["duration_ms"] / many_report["result"]["duration_ms"] \
        == pytest.approx(32, rel=1e-6)
    # Copy work is conserved across all 32 CTAs.
    assert many_report["result"]["declared_service"]["mte2@r0:s0"] == 32 * task_bytes
    assert many_report["result"]["serviced"]["mte2@r0:s0"] == pytest.approx(
        32 * task_bytes, rel=1e-12)


def test_physical_partition_incompatible_geometry_is_rejected_not_averaged():
    op = make_op(params_for([1] * 8, 128, 64))
    with pytest.raises(ValueError, match="resource_ownership_geometry_incompatible"):
        build(op, Geometry(32, {})).program(
            placement_context(op, 32, 24)).programs[0].tile(0)

    with pytest.raises(ValueError, match="active_core_num_exceeds_physical"):
        build(op, Geometry(32, {})).program(placement_context(op, 4, 16))

    geometry = Geometry(32, {}, FULL_BUFFERS,
                        {pool: 24 for pool in FULL_BUFFERS})
    with pytest.raises(ValueError, match="buffer_unit_mapping_incompatible"):
        build(op, geometry).program(
            placement_context(op, 32, 32, FULL_BUFFERS,
                              {pool: 24 for pool in FULL_BUFFERS})
        ).programs[0].tile(0)


def test_tile_addresses_skip_empty_experts_and_cover_each_selected_row_once():
    rows, n, k = [1, 0, 3], 5, 4
    op = make_op(params_for(rows, n, k))
    tiles = list(iter_tiles(op))
    assert [tile["expert_id"] for tile in tiles] == [0, 2]
    assert [tile["w_element_offset"] for tile in tiles] == [0, 2 * k * n]
    assert [tile["x_element_offset"] for tile in tiles] == [0, 1 * k]
    assert [tile["y_element_offset"] for tile in tiles] == [0, 1 * n]

    weight_addresses = set()
    for tile in tiles:
        for row in range(k):
            base = tile["w_element_offset"] + row * tile["weight_row_stride"]
            weight_addresses.update(range(base, base + tile["tile_cols"]))
    assert all(0 <= address < 3 * k * n for address in weight_addresses)
    expert_zero = {address for address in weight_addresses if address < k * n}
    expert_one = {address for address in weight_addresses if k * n <= address < 2 * k * n}
    expert_two = {address for address in weight_addresses if address >= 2 * k * n}
    assert len(expert_zero) == k * n
    assert len(expert_two) == k * n
    assert expert_one == set()  # the empty expert's weight is never read

    x_addresses = set()
    for tile in tiles:
        for row in range(tile["tile_rows"]):
            base = tile["x_element_offset"] + row * k
            x_addresses.update(range(base, base + k))
    assert x_addresses == set(range(4 * k))  # all four input rows are consumed

    y_addresses = set()
    for tile in tiles:
        for row in range(tile["tile_rows"]):
            base = tile["y_element_offset"] + row * n
            y_addresses.update(range(base, base + tile["tile_cols"]))
    assert y_addresses == set(range(4 * n))  # every output row is written
    assert sum(tile["tile_rows"] * k for tile in tiles) == 4 * k
    assert sum(tile["tile_rows"] * tile["tile_cols"] for tile in tiles) == 4 * n


def test_numpy_oracle_proves_compaction_and_rejects_the_raw_break_counterexample():
    expert_rows, n, k = [2, 0, 3], 4, 3
    op = make_op(params_for(expert_rows, n, k))
    x = np.arange(5 * k, dtype=np.int64).reshape(5, k) + 1
    weights = np.stack([
        np.full((k, n), 10, dtype=np.int64),
        np.full((k, n), 100, dtype=np.int64),
        np.full((k, n), 1000, dtype=np.int64),
    ])

    # Independent semantic-histogram reference: skip the empty bin, keep ids.
    y_hist = np.zeros((5, n), dtype=np.int64)
    offset, used = 0, []
    for expert_id, count in ((0, 2), (1, 0), (2, 3)):
        if count == 0:
            continue
        y_hist[offset:offset + count] = x[offset:offset + count] @ weights[expert_id]
        used.append((expert_id, offset, count))
        offset += count
    assert used == [(0, 0, 2), (2, 2, 3)]
    assert offset == 5

    # Independent tile-level reconstruction from the emitted addresses.
    y_tiles = np.zeros((5, n), dtype=np.int64)
    for tile in iter_tiles(op):
        rows = tile["tile_rows"]
        cols = tile["tile_cols"]
        x0 = tile["x_element_offset"] // k
        n0 = tile["n_index"] * SINGLE_N
        a = x[x0:x0 + rows, :]
        b = weights[tile["expert_id"]][:, n0:n0 + cols]
        y0 = tile["y_element_offset"] // n
        y_tiles[y0:y0 + rows, n0:n0 + cols] += a @ b
    assert np.array_equal(y_hist, y_tiles)
    assert np.array_equal(y_hist[:2], x[:2] @ weights[0])
    assert np.array_equal(y_hist[2:], x[2:] @ weights[2])
    assert np.all(np.any(y_hist != 0, axis=1))

    # Literal kernel semantics would truncate at the zero pair.  That is a
    # different output and must never be the semantic histogram adapter.
    raw = interpret_raw_group_list(((0, 2), (1, 0), (2, 3)), n=n, k=k)
    assert raw["processed_rows"] == 2
    y_raw = np.zeros((5, n), dtype=np.int64)
    offset = 0
    for entry in raw["processed"]:
        y_raw[offset:offset + entry["rows"]] = (
            x[offset:offset + entry["rows"]] @ weights[entry["expert_id"]])
        offset += entry["rows"]
    assert not np.array_equal(y_hist, y_raw)
    assert np.array_equal(y_raw[2:], np.zeros((3, n), dtype=np.int64))

    plan = build(op, Geometry(32, {}, {}))
    assert plan.work.matrix["bf16"] == 2 * 5 * n * k
    assert plan.work.matrix["bf16"] != 2 * raw["processed_rows"] * n * k
    # Expert skew is retained per output tile, not replaced by a mean row count.
    assert [tile["flops"] for tile in iter_tiles(op)] == [2 * 2 * n * k, 2 * 3 * n * k]
    assert max(tile["flops"] for tile in iter_tiles(op)) != 2 * 2.5 * n * k


def test_every_tile_stage_is_pinned_to_its_physical_core_and_buffers_are_per_unit():
    rows, n, k = [4, 4, 4], 64, 32
    units = 4
    buffers = dict(FULL_BUFFERS)
    buffer_units = {pool: units for pool in buffers}
    geometry = Geometry(units, {"mte2": units, "mte3": units}, buffers, buffer_units)
    op = make_op(params_for(rows, n, k))
    tiling = host_tiling(op, geometry)
    allocations = tiling["allocation_bytes"]
    names = {"cube": "matrix", "hbm": "memory", "mte2": "read", "mte3": "write"}
    pool_names = {pool: f"{pool}@r0:s0" for pool in buffers}
    context = RecipeContext(
        op, {}, names, {"matrix": 1.0, "memory": 1e12, "read": 1e12, "write": 1e12},
        units, pool_names=pool_names,
        resource_units={"cube": units, "mte2": units, "mte3": units},
        compute_peaks={"bf16": 1e14},
        pool_units=buffer_units)
    program = build(op, geometry).program(context)
    assert isinstance(program, ParallelProgram)
    # core_num=4 but only three compacted experts have tasks, on cores 0..2.
    assert len(program.programs) == 3
    for core, child in zip((0, 1, 2), program.programs):
        load, compute, store = child.tile(0)
        assert load.placement == {"read": (core,)}
        assert compute.placement == {"matrix": (core,)}
        assert store.placement == {"write": (core,)}
        assert len(load.acquire) == 1 and load.acquire[0].units == (core,)
        assert load.acquire[0].size == allocations["l1"]
        assert {item.name for item in compute.acquire} == {
            f"l0a{core}", f"l0b{core}", f"l0c{core}"}
        assert {item.units for item in compute.acquire} == {(core,)}
        assert {item.name for item in compute.free} == {
            f"l1{core}", f"l0a{core}", f"l0b{core}"}
        assert len(store.free) == 1 and store.free[0].name == f"l0c{core}"
        assert store.free[0].units == (core,)

    case = hardware(units=units, split_copy=True, buffers=buffers,
                    buffer_units=buffer_units)
    report = verify_operator(case, op, build)
    assert report["status"] == "passed"
    row = report["result"]
    assert report["work_mismatches"] == {}
    peak = row["peak_buffers_per_unit"]
    for pool in ("l1", "l0a", "l0b", "l0c"):
        active_units = [key for key in peak if key.startswith(f"{pool}@")]
        assert len(active_units) == 3
        assert {peak[key] for key in active_units} == {allocations[pool]}
        # peak_allocated is per-unit, never the sum over the three units.
        assert row["peak_buffers"][f"{pool}@r0:s0"] == allocations[pool]
    assert not any(row["remaining_buffers"].values())

    # The same per-unit residency model on a skewed, tail-bearing histogram.
    skew = make_op(params_for([129, 0, 7], 128, 128))
    skew_geometry = Geometry(32, {"mte2": 32, "mte3": 32}, buffers,
                             {pool: 32 for pool in buffers})
    skew_tiling = host_tiling(skew, skew_geometry)
    skew_case = hardware(units=32, split_copy=True, buffers=buffers,
                         buffer_units={pool: 32 for pool in buffers})
    skew_report = verify_operator(skew_case, skew, build)
    assert skew_report["status"] == "passed"
    skew_row = skew_report["result"]
    assert [tile["tile_rows"] for tile in iter_tiles(skew)] == [128, 1, 7]
    assert skew_row["peak_buffers_per_unit"]["l1@r0:s0:unit0"] == (
        skew_tiling["allocation_bytes"]["l1"])
    assert skew_row["peak_buffers_per_unit"]["l0c@r0:s0:unit2"] == (
        skew_tiling["allocation_bytes"]["l0c"])
    assert len([key for key in skew_row["peak_buffers_per_unit"]
                if key.startswith("l1@")]) == 3
    assert not any(skew_row["remaining_buffers"].values())


def test_buffer_capacity_is_independent_per_physical_unit():
    op = make_op(params_for([4, 4, 4], 64, 32))
    units = 4
    per_unit = hardware(units=units, split_copy=True, buffers=FULL_BUFFERS,
                        buffer_units={pool: units for pool in FULL_BUFFERS})
    shared = hardware(units=units, split_copy=True, buffers=FULL_BUFFERS,
                      buffer_units={pool: 1 for pool in FULL_BUFFERS})
    independent = verify_operator(per_unit, op, build)
    one_unit = verify_operator(shared, op, build)
    assert independent["status"] == "passed" and one_unit["status"] == "passed"
    l1 = host_tiling(op, Geometry(units, {}, FULL_BUFFERS,
                                  {pool: units for pool in FULL_BUFFERS}))[
        "allocation_bytes"]["l1"]
    # With four physical units, each of the three concurrent tile loads uses
    # its own capacity.  With one shared unit, the same loads must queue.
    assert independent["result"]["peak_buffers"]["l1@r0:s0"] == l1
    assert one_unit["result"]["peak_buffers"]["l1@r0:s0"] > l1
    assert one_unit["result"]["peak_buffers"]["l1@r0:s0"] <= FULL_BUFFERS["l1"]
    assert not any(independent["result"]["remaining_buffers"].values())
    assert not any(one_unit["result"]["remaining_buffers"].values())


def test_pool_allocation_mirrors_physical_partition_and_shared_units_contend():
    rows, n, k = [1] * 8, 128, 64
    op = make_op(params_for(rows, n, k))
    panel = host_tiling(op, Geometry(32, {}, FULL_BUFFERS))["allocation_bytes"]["l1"]
    assert panel == 262144

    for pool_units in (8, 16, 32, 64):
        buffer_units = {pool: pool_units for pool in FULL_BUFFERS}
        geometry = Geometry(32, {}, FULL_BUFFERS, buffer_units)
        program = build(op, geometry).program(
            placement_context(op, 32, 32, FULL_BUFFERS, buffer_units))
        assert len(program.programs) == 8
        for core, child in enumerate(program.programs):
            load, compute, store = child.tile(0)
            expected = reference_partition(core, 32, pool_units)
            assert load.acquire[0].units == expected
            assert {item.units for item in compute.acquire} == {expected}
            assert store.free[0].units == expected

        case = hardware(units=32, split_copy=True, buffers=FULL_BUFFERS,
                        buffer_units=buffer_units,
                        rates={"memory": 1e20, "vector": 1e20,
                               "read": 1e20, "write": 1e20},
                        compute_peaks=dict.fromkeys(
                            {"bf16": 1e20, "fp16": 1e20, "fp32": 1e20,
                             "fp8": 1e20, "int8": 1e20, "tf32": 1e20}, 1e20),
                        resource_units={"vector": 32, "mte2": 32, "mte3": 32})
        report = verify_operator(case, op, build)
        assert report["status"] == "passed"
        peak = report["result"]["peak_buffers_per_unit"]
        l1_keys = {key for key in peak if key.startswith("l1@")}
        expected_units = {unit for core in range(8)
                          for unit in reference_partition(core, 32, pool_units)}
        assert {int(key.rsplit("unit", 1)[1]) for key in l1_keys} == expected_units
        for unit in expected_units:
            sharing = sum(1 for core in range(8)
                          if unit in reference_partition(core, 32, pool_units))
            assert peak[f"l1@r0:s0:unit{unit}"] == min(
                sharing * panel, FULL_BUFFERS["l1"])
        assert not any(report["result"]["remaining_buffers"].values())


def test_public_execution_conserves_declared_work_for_unified_and_split_copy_models():
    for case in cases():
        op = make_op(case["params"])
        expected_rows = sum(case["params"]["expert_rows"])
        for split in (False, True):
            report = verify_operator(hardware(split_copy=split), op, build)
            row = report["result"]
            assert report["status"] == "passed"
            assert report["finite_completion"] and report["buffers_released"]
            assert report["work_mismatches"] == {}
            assert row["declared_service"].keys() == row["serviced"].keys()
            for key, amount in row["declared_service"].items():
                assert row["serviced"][key] == pytest.approx(amount, rel=1e-9)
            assert row["operator"]["local_shape"][0] == expected_rows
            assert row["operator"]["matrix_flops"] == {"bf16": row["operator"]["flops"]}
            if split:
                assert set(row["serviced"]) >= {"matrix@r0:s0", "memory@r0:s0",
                                                "mte2@r0:s0", "mte3@r0:s0"}
            else:
                assert set(row["serviced"]) == {"matrix@r0:s0", "memory@r0:s0"}


def test_sparse_expert_ids_keep_the_declared_weight_bank_extent():
    op = make_op(params_for([2, 3], 8, 4, group_ids=[1, 0], weight_experts=4))
    meta = dict(op.metadata)
    assert meta["weight_experts"] == 4
    assert meta["kernel_group_list"] == ((1, 2), (0, 3), (0, 0), (0, 0))
    assert meta["raw_group_list_active"] == ((1, 2), (0, 3))
    assert meta["raw_group_list_extent"] == 4
    assert meta["raw_group_list_padding_entries"] == 2
    assert meta["stop_index"] == 2
    assert meta["stop_reason"] == "split_value_nonpositive"
    assert meta["weight_shape"] == (4, 4, 8)
    entries = meta["processed_entries"]
    assert [(entry["expert_id"], entry["x_row_offset"], entry["w_element_offset"])
            for entry in entries] == [(1, 0, 1 * 4 * 8), (0, 2, 0)]

    with pytest.raises(ValueError, match="group_ids_must_address_declared_weight_experts"):
        make_op(params_for([1, 2], 8, 4, group_ids=[5, 7]))
    sparse = make_op(params_for([1, 2], 8, 4, group_ids=[5, 7], weight_experts=8))
    sparse_meta = dict(sparse.metadata)
    assert sparse_meta["weight_experts"] == 8
    assert len(sparse_meta["kernel_group_list"]) == 8
    assert sparse_meta["kernel_group_list"][:2] == ((5, 1), (7, 2))
    assert sparse_meta["kernel_group_list"][2:] == ((0, 0),) * 6
    assert sparse_meta["weight_shape"] == (8, 4, 8)


def test_runner_preserves_development_and_heldout_geometry_labels():
    module = importlib.import_module("mindie_sim.prediction.operators.grouped_gemm")
    result = run(module, cases())
    assert result["status"] == "passed"
    assert len(result["cases"]) == 10
    assert {row["split"] for row in result["cases"]} == {"development", "heldout_geometry"}
    assert sum(row["split"] == "development" for row in result["cases"]) == 6
    assert sum(row["split"] == "heldout_geometry" for row in result["cases"]) == 4
    by_id = {case["id"]: case for case in cases()}
    for row in result["cases"]:
        assert row["dispatch_mismatches"] == {}
        assert row["identity_invariant"]
        assert row["finite_completion"] and row["buffers_released"]
        assert row["work_mismatches"] == {}
        assert row["result"]["operator"]["local_shape"][0] == sum(
            by_id[row["case"]]["params"]["expert_rows"])


def test_large_histogram_builds_aggregate_work_and_lazy_per_core_programs():
    rows, n, k = [10_000_000, 1], 7168, 128
    op = make_op(params_for(rows, n, k))
    plan = build(op, Geometry(64, {}, {}))
    total_rows = sum(rows)
    expected_flops = 2 * total_rows * n * k
    assert plan.work.matrix["bf16"] == expected_flops
    names = {"cube": "matrix", "hbm": "memory", "mte2": "read", "mte3": "write"}
    rates = {"matrix": 1.0, "memory": 1e9, "read": 1e9, "write": 1e9}
    context = RecipeContext(op, {}, names, rates, 64,
                            resource_units={"cube": 64, "mte2": 64, "mte3": 64})
    program = plan.program(context)
    assert isinstance(program, ParallelProgram)
    expected_tasks = ((10_000_000 + SINGLE_M - 1) // SINGLE_M
                      + (1 + SINGLE_M - 1) // SINGLE_M) * math.ceil(n / SINGLE_N)
    assert sum(child.count for child in program.programs) == expected_tasks
    assert all(not hasattr(child, "stages") for child in program.programs)


def test_missing_rows_are_rejected_at_the_op_boundary_and_build_guard():
    op = make_op(params_for([1, 0, 3], 128, 64))
    meta = dict(op.metadata)
    assert meta["processed_m"] == meta["input_rows"] == 4
    assert meta["missing_rows"] == 0
    with pytest.raises(ValueError, match="x_rows_must_equal_histogram_sum"):
        make_op(params_for([1, 0, 3], 128, 64, x_rows=1))
    bad_metadata = tuple(sorted({
        **meta, "processed_m": 1, "missing_rows": 3}.items()))
    with pytest.raises(ValueError, match="missing_rows_not_accepted"):
        build(replace(op, metadata=bad_metadata), Geometry(32, {}, {}))


def test_source_identity_matches_the_wave3_manifest_and_current_checkout_when_present():
    manifest = json.loads(MANIFEST_FILE.read_text())
    task = manifest
    expected = {source["path"]: (source["sha256"], source["bytes"])
                for source in task["sources"]}
    op = make_op(cases()[0]["params"])
    identity = dict(op.metadata)["source_identity"]
    assert "not_historical_capture_image" in identity["scope"]
    actual = {item["path"]: (item["sha256"], item["bytes"]) for item in identity["files"]}
    assert actual == expected
    for path, (digest, size) in actual.items():
        source = Path(path)
        if source.exists():
            assert source.stat().st_size == size
            assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
