"""Independent structure, traffic, vector and ownership controls for prefill.

Expected ledgers are re-derived from request geometry, executed causal
rectangles and declared tiling.  Synthetic rates make ordering and ownership
observable; they establish no measured hardware accuracy.
"""
from __future__ import annotations

import math
import resource
import sys
import time

import pytest

from mindie_sim.prediction.attention_prefill import (
    RECIPE_ID,
    StagedGQAPrefill,
    logical_causal_pairs,
)
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


def hardware(*, units=4, buffers=None):
    return {
        "machine": {"name": "synthetic-gqa-prefill", "num_devices": 1,
                    "device": {
                        "memory_bytes": 1 << 30,
                        "compute_units": units,
                        "compute_peaks": {"bf16": 8e12, "fp32": 4e12},
                        "rates": {"memory": 1e9, "vector": 1e9, "mte1": 1e9,
                                  "mte2": 1e9, "mte3": 1e9},
                        "buffers": buffers or {
                            "l1": 1 << 22, "l0a": 1 << 21, "l0b": 1 << 21,
                            "l0c": 1 << 22, "ub": 1 << 22},
                        "buffer_units": {"l1": units, "l0a": units, "l0b": units,
                                         "l0c": units, "ub": 2 * units},
                        "resource_units": {"vector": 2 * units, "mte1": units,
                                           "mte2": units, "mte3": units},
                    }},
        "model": {"name": "geometry", "layers": 1, "hidden_size": 16,
                  "num_heads": 4, "num_kv_heads": 2, "head_dim": 8,
                  "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 4,
                                   "output_tokens": 1}]},
        "runtime": {},
    }


def make_op(requests, *, heads=4, kv_heads=2, dim=8, phase="prefill",
            **metadata_overrides):
    metadata = {
        "requests": tuple(requests), "query_heads": heads, "kv_heads": kv_heads,
        "head_dim": dim, "query_dtype": "bf16", "kv_dtype": "bf16",
        "causal": True, "window": 0, "cp": 1,
    }
    metadata.update(metadata_overrides)
    tokens = sum(query for query, _ in requests)
    length = max((context + query for query, context in requests), default=0)
    shape = (tokens, length, heads, dim)
    return OpSpec("gqa_prefill", "attention", phase, 0, 0,
                  tuple(f"r{index}" for index in range(len(requests))),
                  shape, shape, shape, shape, recipe="flash_attention_prefill",
                  dtype="bf16", metadata=(("gqa_prefill", metadata),))


def expected_ledger(requests, *, heads, kv_heads, dim, query_tile, key_tile,
                    query_bytes=2, kv_bytes=2, include_mte1=False):
    """Closed-form independent ledger from per-head executed rectangles."""
    group = heads // kv_heads
    q_read = kv_read = out_write = flops = vector = mte1 = 0
    tasks = key_loads = logical = physical = 0
    for query, context in requests:
        for query_head in range(heads):
            assert query_head // group < kv_heads
            for row_start in range(0, query, query_tile):
                row_end = min(query, row_start + query_tile)
                rows = row_end - row_start
                horizon = context + row_end
                tasks += 1
                logical += sum(context + token + 1
                               for token in range(row_start, row_end))
                q_read += rows * dim * query_bytes
                if include_mte1:
                    mte1 += rows * dim * query_bytes
                for tile_start in range(0, horizon, key_tile):
                    valid = min(key_tile, horizon - tile_start)
                    key_loads += 1
                    kv_read += 2 * valid * dim * kv_bytes
                    if include_mte1:
                        mte1 += 2 * valid * dim * kv_bytes
                    rect = rows * valid
                    physical += rect
                    flops += 4 * dim * rect
                    vector += 7 * rect + rows * (3 * dim + 10)
                    if include_mte1:
                        mte1 += rect * query_bytes
                out_write += rows * dim * query_bytes
                vector += rows * (dim + 1)
    return {
        "tasks": tasks, "key_loads": key_loads,
        "logical": logical, "physical": physical,
        "q_read": q_read, "kv_read": kv_read, "out_write": out_write,
        "hbm": q_read + kv_read + out_write,
        "mte2": q_read + kv_read, "mte3": out_write,
        "flops": flops, "vector": vector, "mte1": mte1,
    }


def run(op, recipe=None, *, units=4, ready=0.0, external=None, raw_case=None,
        trace_limit=20000):
    recipe = recipe or StagedGQAPrefill()
    case = parse_case(raw_case or hardware(units=units))
    execution = Execution(case, CostModel(case),
                          recipes=RecipeBook({RECIPE_ID: recipe}),
                          trace_limit=trace_limit)
    prepared, _ = execution.prepare(op)
    context = execution._context(prepared, 0, "none")
    start = execution.engine.submit("input_ready", release=ready)
    if external:
        execution.engine.submit("external", external["work"],
                                placement=external["placement"], owner="external")
    end = recipe.lower(context).start(execution.engine, "gqa_prefill", start)
    before = {}
    if ready:
        execution.engine.advance(horizon=ready)
        before = {name: value for name, value in execution.engine.serviced.items()
                  if value}
    execution.engine.advance(until=(end,))
    return execution.engine, end, before, context, recipe


def test_gqa_head_reuse_is_not_flattened_into_one_kv_sweep():
    requests = ((5, 7),)
    # Both variants have the same two KV heads.  A shared decode-style KV
    # sweep would read the same bytes; head-local prefill reloads per Q head.
    narrow = expected_ledger(requests, heads=2, kv_heads=2, dim=8,
                             query_tile=4, key_tile=4)
    wide = expected_ledger(requests, heads=4, kv_heads=2, dim=8,
                           query_tile=4, key_tile=4)
    assert wide["tasks"] == 2 * narrow["tasks"]
    assert wide["kv_read"] == 2 * narrow["kv_read"]
    assert wide["hbm"] - wide["q_read"] - wide["out_write"] == 2 * narrow["kv_read"]

    op = make_op(requests)
    recipe = StagedGQAPrefill(query_tile=4, key_tile=4)
    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(wide["hbm"])
    assert engine.serviced["mte2@r0:s0"] == pytest.approx(wide["mte2"])
    assert engine.serviced["mte3@r0:s0"] == pytest.approx(wide["mte3"])


def test_causal_rectangle_differs_from_logical_triangle():
    op = make_op(((3, 5),), heads=1, kv_heads=1)
    recipe = StagedGQAPrefill(query_tile=4, key_tile=4)
    report = recipe.report(op)
    ledger = expected_ledger(((3, 5),), heads=1, kv_heads=1, dim=8,
                             query_tile=4, key_tile=4, include_mte1=True)
    assert report["logical_pairs"] == logical_causal_pairs(3, 5) == 21
    assert report["executed_rectangle_pairs"] == 3 * 8 == 24
    assert report["causal_masked_pairs"] == 3
    assert report["logical_pairs"] == ledger["logical"]
    assert report["executed_rectangle_pairs"] == ledger["physical"]
    work = recipe.work(op, include_mte1=True)
    # Cube work executes the full rectangle, not the 21 valid pairs.
    assert work.matrix == {"bf16": 4 * 8 * 24}
    assert work.resources["vector"] == ledger["vector"]
    assert work.resources["mte1"] == ledger["mte1"]
    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    assert engine.serviced["matrix@r0:s0"] == pytest.approx(
        work.matrix["bf16"] / 8e12 * 1000)
    assert engine.serviced["vector@r0:s0"] == pytest.approx(ledger["vector"])


def test_batched_ragged_query_tails_preserve_per_request_geometry():
    requests = ((3, 5), (5, 2))
    op = make_op(requests)
    recipe = StagedGQAPrefill(query_tile=4, key_tile=4)
    ledger = expected_ledger(requests, heads=4, kv_heads=2, dim=8,
                             query_tile=4, key_tile=4)
    report = recipe.report(op)
    assert report["tasks"] == ledger["tasks"] == 12
    assert report["key_tile_loads"] == ledger["key_loads"]
    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(ledger["hbm"])
    assert engine.serviced["mte2@r0:s0"] == pytest.approx(ledger["mte2"])
    assert engine.serviced["mte3@r0:s0"] == pytest.approx(ledger["mte3"])
    k_loads = [row for row in engine.trace if row["task"].endswith(":k_load")]
    assert len(k_loads) == ledger["key_loads"]
    # The last key tile is real tail work, never a full padded rectangle.
    assert min(row["demand"]["memory@r0:s0"] for row in k_loads) == 2 * 8 * 2


def test_tiny_query_and_partial_last_key_tile_use_actual_rows():
    op = make_op(((1, 5),), heads=1, kv_heads=1)
    recipe = StagedGQAPrefill(query_tile=1, key_tile=4)
    ledger = expected_ledger(((1, 5),), heads=1, kv_heads=1, dim=8,
                             query_tile=1, key_tile=4, include_mte1=True)
    report = recipe.report(op)
    assert report["tasks"] == 1
    assert report["logical_pairs"] == report["executed_rectangle_pairs"] == 6
    assert report["key_tile_loads"] == 2
    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(ledger["hbm"])
    assert engine.serviced["mte1@r0:s0"] == pytest.approx(ledger["mte1"])
    loads = [row["demand"]["memory@r0:s0"] for row in engine.trace
             if row["task"].endswith(":k_load")]
    assert loads == [4 * 8 * 2, 2 * 8 * 2]


def test_no_service_before_nonzero_ready():
    op = make_op(((2, 9),), heads=2, kv_heads=1)
    engine, end, before, _, _ = run(
        op, StagedGQAPrefill(query_tile=4, key_tile=4), ready=1.25)
    assert before == {}
    assert end.time >= 1.25


def test_pinned_task_cannot_borrow_inactive_owner_supply():
    op = make_op(((1, 16),), heads=1, kv_heads=1)
    recipe = StagedGQAPrefill(query_tile=1, key_tile=16)
    baseline, base_end, _, _, _ = run(op, recipe, units=4)
    vector = "vector@r0:s0"
    external = {"work": {vector: 1e7},
                "placement": {vector: (2, 3, 4, 5, 6, 7)}}
    engine, end, _, _, _ = run(op, recipe, units=4, external=external)
    assert end.time == pytest.approx(base_end.time, rel=1e-12, abs=1e-12)
    assert engine.serviced[vector] > 0
    # Only core 0 has a task; three other owners could not supply it.
    assert baseline.peak_tasks > 0


def test_live_tasks_programs_and_buffers_are_released():
    op = make_op(((3, 17),), heads=2, kv_heads=1)
    engine, end, _, _, _ = run(
        op, StagedGQAPrefill(query_tile=2, key_tile=4), units=2)
    assert end.done
    assert not engine._tasks
    assert not engine._promises
    assert not engine.allocations
    assert all(value == 0 for value in engine.allocated.values())


@pytest.mark.parametrize("overrides", (
    {"window": 1}, {"cp": 2}, {"query_heads": 3}, {"causal": False},
))
def test_unsupported_prefill_domain_is_rejected(overrides):
    op = make_op(((1, 8),), **overrides)
    with pytest.raises(ModelingError):
        StagedGQAPrefill().report(op)


def test_declared_decode_phase_is_not_accepted():
    op = make_op(((1, 8),), phase="decode")
    with pytest.raises(ModelingError, match="prefill_phases"):
        StagedGQAPrefill().work(op)


def test_bounded_real_q8192_h6_kv2_d256_run():
    query, context = 8192, 0
    op = make_op(((query, context),), heads=6, kv_heads=2, dim=256)
    recipe = StagedGQAPrefill()
    report = recipe.report(op)
    assert report["tasks"] == 6 * math.ceil(query / 64) == 768
    raw = hardware(units=20, buffers={
        "l1": 524288, "l0a": 65536, "l0b": 65536,
        "l0c": 131072, "ub": 196608,
    })
    started = time.perf_counter()
    engine, end, _, _, _ = run(op, recipe, trace_limit=1000, raw_case=raw)
    elapsed = time.perf_counter() - started
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak *= 1024  # Linux reports KiB; macOS reports bytes.
    assert end.done
    assert elapsed < 30.0
    assert peak < 512 * 1024 * 1024
    assert engine.summary()["live_tasks"] == 0
    assert engine.summary()["live_programs"] == 0
    assert not any(engine.allocated.values())
    # Independent closed form for 128 query tiles, each head reloading KV:
    # sum(horizon)=64*(1+...+128); Q and output are each read/written once.
    expected_hbm = 6 * (2 * query * 256 * 2
                       + 2 * 256 * 2 * 64 * 128 * 129 // 2)
    assert engine.serviced["memory@r0:s0"] == pytest.approx(expected_hbm)


def test_resident_query_and_probability_must_fit_together():
    raw = hardware(buffers={
        "l1": 524288, "l0a": 65536, "l0b": 65536,
        "l0c": 131072, "ub": 196608,
    })
    op = make_op(((256, 0),), heads=4, kv_heads=2, dim=256)
    # Individually Q and P fit 64 KiB. Together they cannot; previously this
    # entered the event loop and stalled permanently instead of stating domain.
    with pytest.raises(ModelingError, match="resident_working_set_exceeds_pool:l0a"):
        run(op, StagedGQAPrefill(query_tile=128, key_tile=128), raw_case=raw)


def test_scale_and_mask_wait_for_qk_and_online_waits_for_pv():
    op = make_op(((2, 6),), heads=1, kv_heads=1)
    engine, end, _, _, _ = run(
        op, StagedGQAPrefill(query_tile=2, key_tile=4), ready=5.0)
    rows = {row["task"]: row for row in engine.trace}
    for name, row in rows.items():
        if name.endswith(":qk"):
            assert "vector@r0:s0" not in row["demand"]
            softmax = rows[name[:-2] + "softmax"]
            assert softmax["start_ms"] >= row["end_ms"]
            assert softmax["demand"]["vector@r0:s0"] == 7 * 2 * 4 + 2
        elif name.endswith(":pv"):
            update = rows[name[:-2] + "online_update"]
            assert update["start_ms"] >= row["end_ms"]
    assert end.done


def test_missing_private_copy_service_is_visible_separately_from_startup():
    op = make_op(((2, 7),), heads=1, kv_heads=1)
    report = StagedGQAPrefill().report(
        op, resource_roles={"hbm", "cube", "vector"})
    assert set(report["unmodeled_transfer_roles"]) == {
        "matrix_input_copy", "matrix_operand_copy", "output_store"}
    assert report["q_bytes"] > 0 and report["kv_bytes"] > 0
