"""Independent structure, traffic, vector and ownership controls for prefill.

Expected ledgers are re-derived from request geometry, executed causal
rectangles and declared tiling.  Synthetic rates make ordering and ownership
observable; they establish no measured hardware accuracy.
"""
from __future__ import annotations

import math
try:
    import resource
except ImportError:
    resource = None
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
                for tile_start in range(0, horizon, key_tile):
                    valid = min(key_tile, horizon - tile_start)
                    key_loads += 1
                    if include_mte1:
                        # Every S2 loop reloads the resident Q tile L1->L0A.
                        mte1 += rows * dim * query_bytes
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
        trace_limit=20000, buffers=None):
    recipe = recipe or StagedGQAPrefill()
    case = parse_case(raw_case or hardware(units=units, buffers=buffers))
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
    assert end.done
    # This is the complete large-geometry execution control. Its 90 s budget
    # follows the local end-to-end tier; CPU throughput varies by runner.
    # The live task bound, rather than host speed, guards against expansion.
    assert elapsed < 90.0
    assert engine.peak_tasks < 4096
    if resource is not None:
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if peak < (1 << 20):
            peak *= 1024  # Linux reports KiB; macOS reports bytes.
        assert peak < 512 * 1024 * 1024
    assert engine.summary()["live_tasks"] == 0
    assert engine.summary()["live_programs"] == 0
    assert not any(engine.allocated.values())
    # Independent closed form for 128 query tiles, each head reloading KV:
    # sum(horizon)=64*(1+...+128); Q and output are each read/written once.
    expected_hbm = 6 * (2 * query * 256 * 2
                       + 2 * 256 * 2 * 64 * 128 * 129 // 2)
    assert engine.serviced["memory@r0:s0"] == pytest.approx(expected_hbm)


def test_query_is_l1_resident_and_l0a_uses_shared_microtiles():
    raw = hardware(buffers={
        "l1": 524288, "l0a": 65536, "l0b": 65536,
        "l0c": 131072, "ub": 196608,
    })
    op = make_op(((256, 0),), heads=4, kv_heads=2, dim=256)
    # Arch35 keeps Q in L1 and streams it through two shared 32 KiB L0A
    # microtile buffers.  A 64 KiB Q panel is therefore not a resident L0A
    # working set and must not be rejected as Q+P.
    engine, end, _, _, _ = run(
        op, StagedGQAPrefill(query_tile=128, key_tile=128), raw_case=raw)
    assert end.done
    assert not engine.allocations
    assert not any(engine.allocated.values())


def test_native_fia_selects_arch35_cube_block_not_cv_diff_head_rule():
    from dataclasses import replace

    op = make_op(((8192, 0),), heads=8, kv_heads=1, dim=128)
    native = replace(op, metadata=op.metadata + (
        ("native_operator", "FusedInferAttentionScore"),))
    native_report = StagedGQAPrefill().report(native)
    assert native_report["query_tile"] == 128
    # D256 is the non-quant GQA case excluded from the registered templates;
    # its selected V4 branch has SOuter64*CV_RATIO=128, not the CVDiff 64.
    wide = make_op(((8192, 0),), heads=4, kv_heads=1, dim=256)
    wide_native = replace(wide, metadata=wide.metadata + (
        ("native_operator", "FusedInferAttentionScore"),))
    assert StagedGQAPrefill().report(wide_native)["query_tile"] == 128
    # Generic (unbound) GQA keeps the CVDiff source contract for callers that
    # have not selected FusedInferAttentionScore.
    assert StagedGQAPrefill().report(wide)["query_tile"] == 64


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


def test_reused_key_panel_templates_preserve_tail_events_and_resource_service(monkeypatch):
    import mindie_sim.prediction.attention_prefill as module

    op = make_op(((9, 13), (7, 2)), heads=4, kv_heads=2, dim=8)
    recipe = StagedGQAPrefill(query_tile=4, key_tile=5, slots=2)
    cached, cached_end, _, _, _ = run(op, recipe, ready=7, trace_limit=10000)
    monkeypatch.setattr(module, "lru_cache", lambda **_: lambda fn: fn)
    expanded, expanded_end, _, _, _ = run(op, recipe, ready=7, trace_limit=10000)
    assert cached_end.time == expanded_end.time
    assert cached.serviced == expanded.serviced
    assert list(cached.trace) == list(expanded.trace)
    assert cached.summary()["live_tasks"] == cached.summary()["live_programs"] == 0
    assert not any(cached.allocated.values())


def test_q_l1_to_l0a_is_paid_per_key_tile_not_once_per_call():
    """Arch35 reloads the resident Q tile for every S2 loop.

    The frozen candidate charged Q L1->L0A once (16,777,216 B for 8192Q),
    missing 528,482,304 B.  Q GM->L1 remains one transfer per task.
    """
    from dataclasses import replace

    for heads, dim in ((8, 128), (4, 256)):
        op = make_op(((8192, 0),), heads=heads, kv_heads=1, dim=dim)
        native = replace(op, metadata=op.metadata + (
            ("native_operator", "FusedInferAttentionScore"),))
        report = StagedGQAPrefill().report(native)
        assert report["query_tile"] == 128
        assert report["tasks"] == heads * 64
        assert report["q_bytes"] == 16_777_216
        assert report["q_l1_l0a_bytes"] == 545_259_520
        assert report["q_l1_l0a_bytes"] - report["q_bytes"] == 528_482_304

    # A q < cube block still pays once per actual key tile, including the
    # final partial key tile; no logical-pair or padded-horizon shortcut.
    op = make_op(((65, 64),), heads=4, kv_heads=1, dim=256)
    native = replace(op, metadata=op.metadata + (
        ("native_operator", "FusedInferAttentionScore"),))
    report = StagedGQAPrefill().report(native)
    q_bytes = 4 * 65 * 256 * 2
    assert report["q_bytes"] == q_bytes
    # Native M tiles accumulate time x G4 into 128-row tiles: only 3 M tiles,
    # each with one S2 tile at q65/kv129; per-head 4x2 is not the native order.
    assert report["key_tile_loads"] == 4
    # M0/M1 are full 128-row tiles; M2 is the real 4-row tail with two S2
    # tiles (horizon 129).  Q L1->L0A follows the executed rows, not a full
    # padded 128-row tile.
    assert report["q_l1_l0a_bytes"] == 2 * 128 * 256 * 2 + 2 * 4 * 256 * 2


def test_source_owner_plan_matches_arch35_split_core_boundaries():
    from dataclasses import replace

    def native(q, kv, heads=8, dim=128):
        op = make_op(((q, kv - q),), heads=heads, kv_heads=1, dim=dim)
        return replace(op, metadata=op.metadata + (
            ("native_operator", "FusedInferAttentionScore"),))

    q8192 = StagedGQAPrefill(query_tile=128, key_tile=128).owner_plan(
        native(8192, 8192), cores=32)
    assert q8192["total"] == 16640
    assert set(q8192["loads"]) == {520}
    assert q8192["fd_splits"] == 24

    q4096 = StagedGQAPrefill(query_tile=128, key_tile=128).owner_plan(
        native(4096, 4096), cores=32)
    assert q4096["total"] == 4224
    assert set(q4096["loads"]) == {132}
    assert q4096["fd_splits"] == 24

    q16 = StagedGQAPrefill(query_tile=128, key_tile=128).owner_plan(
        native(8192, 8192), cores=16)
    assert set(q16["loads"]) == {1040}
    assert q16["fd_splits"] == 11

    long = StagedGQAPrefill(query_tile=128, key_tile=128).owner_plan(
        native(8192, 131072), cores=32)
    assert set(long["loads"]) == {15880}
    assert long["fd_splits"] == 31

    small = StagedGQAPrefill(query_tile=128, key_tile=128).owner_plan(
        native(1024, 1024), cores=32)
    assert small["total"] == 288 and small["fd_splits"] == 0


def test_fd_workspace_is_reported_for_split_owners():
    from dataclasses import replace

    op = make_op(((8192, 0),), heads=8, kv_heads=1, dim=128)
    native = replace(op, metadata=op.metadata + (
        ("native_operator", "FusedInferAttentionScore"),))
    report = StagedGQAPrefill().report(native, cores=32)
    assert report["fd_splits"] == 24
    assert set(report["owner_s2_loads"]) == {520}
    assert report["fd_partial_write_bytes"] == 24 * 128 * 128 * 4
    assert report["fd_partial_read_bytes"] == report["fd_partial_write_bytes"]
    assert report["fd_lse_bytes"] == 24 * 128 * 2 * 4
    assert report["fd_reduce_ops"] == 24 * 128 * 130


def test_fd_merge_is_real_engine_work_not_a_report_row():
    from dataclasses import replace

    op = make_op(((4096, 0),), heads=8, kv_heads=1, dim=128)
    native = replace(op, metadata=op.metadata + (
        ("native_operator", "FusedInferAttentionScore"),))
    raw = hardware(units=32)
    device = raw["machine"]["device"]
    device["rates"]["mte2_aic"] = 1e9
    device["rates"]["mte2_aiv"] = 1e9
    device["resource_units"]["mte2_aic"] = 32
    device["resource_units"]["mte2_aiv"] = 64
    engine, end, _, _, _ = run(
        native, StagedGQAPrefill.arch35_noquant_candidate(),
        raw_case=raw, trace_limit=200000)
    assert end.done
    fd = [row for row in engine.trace if ":fd" in row["task"]]
    stores = [row for row in engine.trace if row["task"].endswith(":final_store")]
    assert len(fd) == 48  # 24 split-M x two 64-row AIV fragments
    assert len(stores) == 280  # 256 full tasks + 24 extra split writers
    assert engine.serviced["mte2_aiv@r0:s0"] == pytest.approx(3_194_880)
    assert all(row["demand"] for row in fd)
    # The merge reads/writes real Engine bytes, not a zero-work report row.
    assert engine.serviced["mte2_aiv@r0:s0"] > 0
    assert engine.serviced["mte2_aic@r0:s0"] > 0
    assert engine.serviced["mte3@r0:s0"] > 0
    assert engine.summary()["live_tasks"] == 0


def test_native_m_tiles_preserve_time_group_tail_geometry():
    from dataclasses import replace

    def native(q):
        op = make_op(((q, 0),), heads=8, kv_heads=1, dim=128)
        return replace(op, metadata=op.metadata + (
            ("native_operator", "FusedInferAttentionScore"),))

    q129 = StagedGQAPrefill().owner_plan(native(129), cores=32)
    assert q129["total"] == 10
    assert sum(len(x) for x in q129["tasks"]) == 9
    assert [row for row in q129["tasks"] if row] == [(i,) for i in range(9)]
    assert StagedGQAPrefill().report(native(129))["tasks"] == 9

    q257 = StagedGQAPrefill().owner_plan(native(257), cores=32)
    assert q257["total"] == 27
    assert sum(len(x) for x in q257["tasks"]) == 17
    # Host split_core_v2 right-down search uses 13 cores, not one core per M row.
    assert [row for row in q257["tasks"] if row] == [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,),
    ]
    assert StagedGQAPrefill().report(native(257))["tasks"] == 17

    q1024 = StagedGQAPrefill().owner_plan(native(1024), cores=32)
    assert q1024["total"] == 288
    assert sum(len(x) for x in q1024["tasks"]) == 64
    assert q1024["fd_splits"] == 0
    active = [row for row in q1024["tasks"] if row]
    assert len(active) == 32
    assert active[0] == tuple(range(8))
    assert active[1] == (8, 9, 10, 11)
    assert active[-1] == (62, 63)
    assert [index for row in active for index in row] == list(range(64))


def test_nofd_ragged_owner_rows_follow_host_search():
    """Unseen S1/S2 pairs.  Groups are the host core ends, not a timing table."""
    from dataclasses import replace

    from mindie_sim.prediction import attention_prefill

    def native(query, kv, heads=8, kv_heads=1):
        op = make_op(((query, kv - query),), heads=heads, kv_heads=kv_heads, dim=128)
        return replace(op, metadata=op.metadata + (
            ("native_operator", "FusedInferAttentionScore"),))

    def native_requests(pairs, heads=8, kv_heads=1):
        op = make_op(tuple((query, kv - query) for query, kv in pairs),
                     heads=heads, kv_heads=kv_heads, dim=128)
        return replace(op, metadata=op.metadata + (
            ("native_operator", "FusedInferAttentionScore"),))

    short = StagedGQAPrefill().owner_plan(native(40, 96), cores=32)
    assert short["fd_splits"] == 0
    assert [row for row in short["tasks"] if row] == [(0,), (1,), (2,)]

    wide = StagedGQAPrefill().owner_plan(native(63, 200, heads=4), cores=32)
    assert wide["fd_splits"] == 0
    assert [row for row in wide["tasks"] if row] == [(0,), (1,)]

    mixed = StagedGQAPrefill().owner_plan(native(300, 400), cores=32)
    assert [row for row in mixed["tasks"] if row] == [
        (0, 1), (2, 3), (4, 5), (6, 7),
        (8,), (9,), (10,), (11,), (12,), (13,), (14,), (15,), (16,), (17,), (18,),
    ]

    # Two KV heads stay disjoint whole rows.  The host search owns both heads.
    spec = attention_prefill._spec(native(129, 129, kv_heads=2), StagedGQAPrefill())
    plan = attention_prefill.owner_s2_plan(spec, 32)
    native_rows = tuple(attention_prefill._native_tasks(spec))
    flat = [index for row in plan["tasks"] for index in row]
    assert sorted(flat) == [index for index, _task in native_rows]
    assert len(flat) == len(set(flat)) == 10
    assert {task.kv_head for _index, task in native_rows} == {0, 1}
    assert plan["fd_splits"] == 0

    # S2 total 306 is above cores*tolerance and the host still keeps FD off.
    wide_square = StagedGQAPrefill().owner_plan(native(1056, 1056), cores=32)
    wide_active = [row for row in wide_square["tasks"] if row]
    assert wide_square["fd_splits"] == 0
    assert wide_square["loads"] == (
        10, 10, 10, 9, 9, 8, 8, 8, 8, 10, 10, 10, 10, 6, 6, 6, 6,
        12, 12, 7, 7, 7, 7, 14, 14, 8, 8, 16, 16, 16, 18, 0)
    assert wide_active[0] == tuple(range(9))
    assert wide_active[-1] == (64, 65)
    assert [index for row in wide_active for index in row] == list(range(66))
    assert min(wide_square["loads"]) >= 0

    pair = StagedGQAPrefill().owner_plan(
        native_requests(((130, 130), (193, 193)), heads=8, kv_heads=2), cores=32)
    assert pair["fd_splits"] == 0
    assert [row for row in pair["tasks"] if row] == [
        (0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13),
        (14,), (15,), (16, 17), (18, 19), (20,), (21,), (22,), (23,),
    ]
    assert sum(pair["loads"]) == pair["total"] == 32
    assert min(pair["loads"]) >= 0

    two = StagedGQAPrefill().owner_plan(
        native_requests(((1024, 1024), (1, 1)), heads=8, kv_heads=2), cores=32)
    two_active = [row for row in two["tasks"] if row]
    assert two["fd_splits"] == 0
    assert len(two_active) == 32
    assert two_active[0] == tuple(range(6))
    assert two_active[-1] == (62, 63, 64, 65)
    assert [index for row in two_active for index in row] == list(range(66))
    assert sum(two["loads"]) == two["total"] == 290
    assert min(two["loads"]) >= 0

    # q1030 keeps flash-decode.  The host cost check uses 17 split rows, and
    # the tail load stays non-negative.  Equal chunks of this total reported 16.
    split = StagedGQAPrefill().owner_plan(native(1030, 1030), cores=32)
    owned = {}
    for core_rows in split["ranges"]:
        for task, lo, hi in core_rows:
            assert hi >= lo
            for tile in range(lo, hi):
                owned[task, tile] = owned.get((task, tile), 0) + 1
    assert split["fd_splits"] == 17
    assert split["loads"] == (
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
        9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 10, 9, 14, 0, 0)
    assert split["total"] == 297
    assert sum(split["loads"]) == 297
    assert min(split["loads"]) >= 0
    assert len(owned) == 297
    assert set(owned.values()) == {1}


def test_native_m_tiles_cover_every_kv_group_and_head_ratio():
    from dataclasses import replace

    def native(q, heads, kv_heads):
        op = make_op(((q, 0),), heads=heads, kv_heads=kv_heads, dim=128)
        return replace(op, metadata=op.metadata + (
            ("native_operator", "FusedInferAttentionScore"),))

    q129 = {
        1: (9, 360_960),
        2: (10, 459_776),
        4: (12, 657_408),
    }
    for kv_heads, (tasks, kv_bytes) in q129.items():
        report = StagedGQAPrefill().report(native(129, 8, kv_heads))
        assert report["tasks"] == tasks
        assert report["kv_bytes"] == kv_bytes
        assert report["logical_pairs"] == 6 * 0 + 8 * logical_causal_pairs(129, 0)

    two_head = StagedGQAPrefill().report(native(16, 8, 2))
    assert two_head["tasks"] == 2
    assert two_head["kv_bytes"] == 16_384
    assert two_head["logical_pairs"] == 8 * logical_causal_pairs(16, 0)

    six = StagedGQAPrefill().report(native(129, 6, 2))
    assert six["tasks"] == 8
    assert six["kv_bytes"] == 395_264
    assert six["logical_pairs"] == 6 * logical_causal_pairs(129, 0) == 50_310


def test_native_s2_preload_overlaps_both_geometries_and_leaves_c2_unselected():
    """Source keeps three S2 tasks resident on D128 and D256.

    The generic double buffer issues QK of tile 2 only after tile 0 retires.
    Native preload issues that QK while tile 0 is still in flight.  The
    reconstruct flag and the D256 GM workspace stay unselected.
    """
    import re
    from dataclasses import replace

    from mindie_sim.prediction import attention_prefill
    from mindie_sim.prediction.execution import Execution

    assert Execution.schedule_replay is False
    assert attention_prefill.source_issue_operand(None) is None
    assert attention_prefill.source_issue_operand(True) == "real"
    assert attention_prefill.source_issue_operand(False) == "causal"
    assert attention_prefill.source_c2_path(128, 128, 128, None) is None
    assert attention_prefill.source_c2_path(256, 128, 128, None) is None
    assert attention_prefill.source_c2_path(256, 128, 128, True) is None
    gm = attention_prefill.source_c2_path(256, 128, 128, False)
    assert gm == {"path": "gm_workspace", "workspace_bytes": 12_582_912,
                  "hbm_bytes": 0}
    assert attention_prefill.source_c2_path(256, 64, 256, False)["path"] == "ub"
    assert attention_prefill.source_c2_path(128, 128, 128, True)["path"] == "ub"
    assert attention_prefill.source_c2_path(128, 64, 256, False)["path"] == "ub"

    def boundary(dim, heads):
        op = make_op(((16 if heads == 8 else 32, 1024 - (16 if heads == 8 else 32)),),
                     heads=heads, kv_heads=1, dim=dim)
        native = replace(op, metadata=op.metadata + (
            ("native_operator", "FusedInferAttentionScore"),))
        spec = attention_prefill._spec(native, StagedGQAPrefill())
        assert spec.s2_preload == 3
        assert spec.slots == 2
        engine, end, _, _, _ = run(native, units=1, trace_limit=20000)
        assert end.done and engine.summary()["live_tasks"] == 0
        found = {}
        pattern = re.compile(r"key_tiles:program:tile(\d+):(qk|online_update)$")
        for row in engine.trace:
            match = pattern.search(row["task"])
            if match:
                found.setdefault(int(match.group(1)), {})[match.group(2)] = (
                    row["start_ms"], row["end_ms"])
        assert found[2]["qk"][0] < found[0]["online_update"][1]
        return end.time

    boundary(128, 8)
    boundary(256, 4)


def test_native_kv_group_tasks_are_distinct_and_conserved():
    from dataclasses import replace
    from mindie_sim.prediction import attention_prefill

    def native(q, heads, kv_heads, dim=128):
        op = make_op(((q, 0),), heads=heads, kv_heads=kv_heads, dim=dim)
        return replace(op, metadata=op.metadata + (
            ("native_operator", "FusedInferAttentionScore"),))

    for q, heads, kv_heads in ((129, 8, 2), (129, 8, 4), (16, 8, 2),
                                (130, 8, 2), (193, 8, 2), (513, 8, 4)):
        op = native(q, heads, kv_heads)
        spec = attention_prefill._spec(op, StagedGQAPrefill())
        tasks = tuple(attention_prefill._iter_tasks(spec))
        keys = [(task.kv_head, task.row_start) for task in tasks]
        assert len(keys) == len(set(keys))
        assert sum(task.rows for task in tasks) == q * heads
        assert {task.kv_head for task in tasks} == set(range(kv_heads))
        assert all(task.heads == heads // kv_heads for task in tasks)


def test_issue_kind_follows_explicit_reconstruct_and_l1_operands_fit():
    """Absent reconstruct selects nothing.  Four descriptors are not four K/V slots.

    The static BF16 reservations are the cube buffer policy.  589824 B is not
    a required L1 total for any of these four templates.
    """
    from dataclasses import replace

    from mindie_sim.prediction import attention_prefill as ap
    from mindie_sim.prediction.attention_prefill_issue import (
        bmm_call, l0c_policy, l1_operand_capacity, source_issue_kind,
    )

    assert Execution.schedule_replay is False
    assert source_issue_kind(None, 128) is None
    assert source_issue_kind(None, 256) is None
    assert source_issue_kind(True, 256) is None
    assert source_issue_kind(True, 128) == "gqa"
    assert source_issue_kind(False, 128) == "simplified"
    assert source_issue_kind(False, 256) == "simplified"

    def native(dim, reconstruct, **overrides):
        op = make_op(((16, 0),), heads=8, kv_heads=1, dim=dim, **overrides)
        extra = (("native_operator", "FusedInferAttentionScore"),)
        if reconstruct is not None:
            extra += (("reconstruct", reconstruct),)
        return replace(op, metadata=op.metadata + extra)

    absent = ap._spec(native(256, None), StagedGQAPrefill())
    assert absent.reconstruct is None and absent.issue_kind is None
    inside = make_op(((16, 0),), heads=8, kv_heads=1, dim=128, reconstruct=False)
    inside = replace(inside, metadata=inside.metadata + (
        ("native_operator", "FusedInferAttentionScore"),))
    nested = ap._spec(inside, StagedGQAPrefill())
    assert nested.issue_kind is None and nested.reconstruct is None
    unregistered = ap._spec(native(256, True), StagedGQAPrefill())
    assert unregistered.reconstruct is True and unregistered.issue_kind is None
    with pytest.raises(ModelingError, match="reconstruct_must_be_bool_or_absent"):
        ap._spec(native(128, 0), StagedGQAPrefill())

    expected = {
        (128, 128, 256): 491520,
        (64, 256, 256): 425984,
        (128, 128, 128): 294912,
        (64, 256, 128): 393216,
    }
    for (s1, s2, dim), total in expected.items():
        cap = l1_operand_capacity(s1, s2, dim)
        assert cap["total"] == total
        assert cap["fits"] and cap["l1_pool"] == 524288
        assert cap["p_count"] == 3
        assert cap["k_count"] == cap["v_count"]
        assert cap["k_count"] in (1, 2)
        assert cap["total"] != 589824
    six, eighteen, three, one = (
        l1_operand_capacity(128, 128, 256),
        l1_operand_capacity(64, 256, 256),
        l1_operand_capacity(128, 128, 128),
        l1_operand_capacity(64, 256, 128),
    )
    assert (six["q_count"], six["q_bytes"], six["k_count"], six["k_bytes"]) == (
        2, 65536, 2, 65536)
    assert (eighteen["q_count"], eighteen["q_bytes"], eighteen["k_count"],
            eighteen["k_bytes"]) == (2, 32768, 1, 131072)
    assert (three["q_count"], three["k_count"], three["k_bytes"]) == (2, 2, 32768)
    assert (one["q_count"], one["q_bytes"], one["k_count"], one["k_bytes"]) == (
        2, 16384, 2, 65536)
    assert l0c_policy(128, 128, 256) == {
        "count": 2, "piece": 131072, "pool": 262144, "total": 262144}
    for shape in ((128, 128, 128), (64, 256, 256), (64, 256, 128)):
        assert l0c_policy(*shape)["count"] == 4
        assert l0c_policy(*shape)["piece"] == 65536
    assert bmm_call(128, 128, 256, 256) == {
        "qk": "IterateBmm1NdL0Split MatmulK<128,128,128> singleK=d",
        "pv": "MatmulN singleN=dV baseN=128",
    }
    assert bmm_call(128, 128, 128, 128) == {
        "qk": "IterateBmm1Nd MatmulBase<128,128,d>",
        "pv": "MatmulFull<128,dV,128>",
    }
    assert bmm_call(64, 256, 256, 256)["qk"].startswith(
        "IterateBmm1NdL0Split MatmulN<64,128,256>")
    assert bmm_call(64, 256, 128, 128)["pv"] == "MatmulBase<128,dV,128>"


def test_simplified_and_gqa_cursors_cross_m_without_four_kv_slots():
    """One task cursor spans M.  K/V slot count stays the operand policy."""
    from dataclasses import replace

    from mindie_sim.prediction import attention_prefill as ap
    from mindie_sim.prediction.attention_prefill_nofd import owner_s2_plan
    from tests.test_sku_pipeline_profiles import A5_BUFFERS

    def native(dim, heads, reconstruct):
        op = make_op(((32, 992),), heads=heads, kv_heads=1, dim=dim)
        extra = (("native_operator", "FusedInferAttentionScore"),)
        if reconstruct is not None:
            extra += (("reconstruct", reconstruct),)
        return replace(op, metadata=op.metadata + extra)

    held = native(256, 4, True)
    assert ap._spec(held, StagedGQAPrefill()).issue_kind is None
    held_engine, held_end, _, _, _ = run(
        held, units=1, trace_limit=4000, buffers=A5_BUFFERS)
    assert held_end.done and held_engine.summary()["live_tasks"] == 0
    assert any("key_tiles:program:tile" in row["task"] for row in held_engine.trace)
    assert all("kind" not in (row.get("metadata") or {}) for row in held_engine.trace)

    def indexed(engine):
        found = {}
        for row in engine.trace:
            meta = row.get("metadata") or {}
            stage = meta.get("stage")
            if not stage:
                continue
            found[(stage, meta["tile"])] = {
                **meta, "start": row["start_ms"], "end": row["end_ms"],
                "task": row["task"],
            }
        return found

    for reconstruct, kind, vec1_step, vec2_step, window in (
            (False, "simplified", 1, 3, 4),
            (True, "gqa", 0, 2, 3)):
        op = native(128, 8, reconstruct)
        spec = ap._spec(op, StagedGQAPrefill())
        assert spec.issue_kind == kind and spec.s2_preload == 3
        plan = owner_s2_plan(spec, 1)
        ranges = plan["ranges"][0]
        assert plan["fd_splits"] == 0 and len(ranges) >= 2
        boundary = ranges[0][2] - ranges[0][1]
        assert boundary >= 8
        engine, end, _, _, _ = run(op, units=1, trace_limit=8000, buffers=A5_BUFFERS)
        assert end.done and engine.trace_dropped == 0
        summary = engine.summary()
        assert summary["live_tasks"] == 0
        assert 4 < summary["peak_live_tasks"] < 80
        found = indexed(engine)
        assert found[("qk", boundary)]["start"] < found[("vec2", boundary - 1)]["start"]
        assert found[("qk", 0)]["window"] == window
        assert found[("pv", 0)]["step"] == 2
        assert found[("pv", 0)]["start"] >= found[("qk", 2)]["end"] - 1e-9
        assert found[("vec1", 0)]["step"] == vec1_step
        assert found[("vec2", 0)]["step"] == vec2_step
        assert found[("vec2", 0)]["start"] >= found[("vec1", 2)]["end"] - 1e-9
        slots = set()
        for (stage, _tile), meta in found.items():
            if stage != "k_load":
                continue
            assert meta["k_count"] == 2 and meta["k_slot"] in (0, 1)
            assert ":l1:k2" not in meta["task"] and ":l1:k3" not in meta["task"]
            slots.add(meta["k_slot"])
        assert slots == {0, 1}
        peak_l1 = max(value for key, value in summary["peak_allocated"].items()
                      if key.startswith("l1"))
        assert peak_l1 <= 524288


def test_key256_cursor_is_hand_segments_not_the_equal_chunk_owner():
    """#1 and #18 stay off the key_tile!=128 owner fallback."""
    from dataclasses import replace

    from mindie_sim.prediction import attention_prefill as ap
    from mindie_sim.prediction.attention_prefill_issue import IssueCursor
    from mindie_sim.prediction.attention_prefill_nofd import _work_from_spec
    from mindie_sim.prediction.cost import CostModel
    from mindie_sim.prediction.program import RecipeBook
    from tests.test_sku_pipeline_profiles import A5_BUFFERS

    def drive(dim, query_tile, key_tile, k_count):
        recipe = StagedGQAPrefill(query_tile=query_tile, key_tile=key_tile)
        op = make_op(((query_tile, 0),), heads=1, kv_heads=1, dim=dim)
        native = replace(op, metadata=op.metadata + (
            ("native_operator", "FusedInferAttentionScore"),
            ("reconstruct", False),
        ))
        spec = ap._spec(native, recipe)
        assert spec.issue_kind == "simplified"
        assert spec.query_tile == query_tile and spec.key_tile == key_tile
        assert _work_from_spec(spec) is None
        case = parse_case(hardware(units=1, buffers=A5_BUFFERS))
        execution = Execution(
            case, CostModel(case), recipes=RecipeBook({RECIPE_ID: recipe}),
            trace_limit=4000)
        prepared, _ = execution.prepare(native)
        context = execution._context(prepared, 0, "none")
        segments = [
            {"rows": query_tile, "horizon": 8 * key_tile, "key_start": 0,
             "key_tiles": 8, "partial": False},
            {"rows": query_tile, "horizon": 8 * key_tile, "key_start": 0,
             "key_tiles": 8, "partial": False},
        ]
        start = execution.engine.submit("ready")
        end = IssueCursor(context, spec, segments, 0).start(
            execution.engine, "hand", start)
        execution.engine.advance(until=(end,))
        assert end.done and execution.engine.trace_dropped == 0
        summary = execution.engine.summary()
        assert summary["live_tasks"] == 0
        assert summary["peak_live_tasks"] < 80
        peak_l1 = max(value for key, value in summary["peak_allocated"].items()
                      if key.startswith("l1"))
        assert peak_l1 <= 524288
        qk, vec2, slots = {}, {}, set()
        for row in execution.engine.trace:
            meta = row.get("metadata") or {}
            if meta.get("stage") == "qk":
                qk[meta["tile"]] = row["start_ms"]
            elif meta.get("stage") == "vec2":
                vec2[meta["tile"]] = row["start_ms"]
            elif meta.get("stage") == "k_load":
                assert meta["k_count"] == k_count
                assert meta["k_slot"] < k_count
                slots.add(meta["k_slot"])
                assert ":l1:k2" not in row["task"] and ":l1:k3" not in row["task"]
        assert qk[8] < vec2[7]
        assert slots == set(range(k_count))

    drive(256, 64, 256, 1)
    drive(128, 64, 256, 2)


def test_source_compatible_set_stays_unselected_on_the_bare_path():
    """D256 class is Simplified and still open.  D128 stays mixed.

    The set is the registered range.  Bare issue_kind stays null, and an
    explicit reconstruct flag does not collapse #6 versus #18.
    """
    from dataclasses import replace

    from mindie_sim.prediction import attention_prefill as ap
    from mindie_sim.prediction.attention_prefill_issue import (
        source_compatible_alternatives,
    )

    d256 = source_compatible_alternatives(256, (8192,))
    assert d256["selected"] is None and d256["class"] == "simplified"
    assert d256["reconstruct"] == (False,) and d256["reason"] == "topology_open"
    assert [row["config"] for row in d256["candidates"]] == [6, 18]
    assert [row["c2"] for row in d256["candidates"]] == ["gm_workspace", "ub"]
    assert d256["candidates"][0]["workspace_bytes"] == 12_582_912
    assert d256["candidates"][0]["hbm_bytes"] == 0
    for row in d256["candidates"]:
        assert row["fd_s1_out"] == ((False, False), (True, False), (False, True))
        assert row["struct"] == "FlashAttentionScoreSimplifiedTilingData"
    d128 = source_compatible_alternatives(128, (2, 1))
    assert d128["selected"] is None and d128["class"] == "mixed"
    assert d128["reconstruct"] == (False, True)
    assert [row["config"] for row in d128["candidates"]] == [1, 1, 3, 3]
    assert source_compatible_alternatives(256, (1,))["reason"] == (
        "q_len_1_incre_flash_attention")
    assert source_compatible_alternatives(256, (8192,), rope=True)["candidates"] == ()
    assert source_compatible_alternatives(256, (8192,), out_dtype="int8")["class"] is None
    assert source_compatible_alternatives(64, (8192,))["reason"] == "head_dim_unregistered"
    from mindie_sim.prediction.attention_prefill_issue import source_issue_kind
    assert source_issue_kind(None, 128) is None
    assert source_issue_kind(None, 256) is None
    assert source_issue_kind(True, 256) is None

    def native(requests, dim, reconstruct=None, **overrides):
        op = make_op(requests, heads=4, kv_heads=1, dim=dim, **overrides)
        extra = (("native_operator", "FusedInferAttentionScore"),)
        if reconstruct is not None:
            extra += (("reconstruct", reconstruct),)
        return replace(op, metadata=op.metadata + extra)

    bare = ap._spec(native(((32, 0),), 256), StagedGQAPrefill())
    assert bare.issue_kind is None and bare.reconstruct is None
    report = StagedGQAPrefill().report(native(((32, 0),), 256))
    assert report["issue_kind"] is None
    assert report["source_compatible"]["selected"] is None
    assert report["source_compatible"]["class"] == "simplified"
    mixed = StagedGQAPrefill().report(native(((32, 0),), 128))
    assert mixed["issue_kind"] is None
    assert mixed["source_compatible"]["class"] == "mixed"
    held = ap._spec(native(((32, 0),), 256, False), StagedGQAPrefill())
    assert held.issue_kind == "simplified"
    flagged = StagedGQAPrefill().report(native(((32, 0),), 256, False))
    assert flagged["source_compatible"]["selected"] is None
    assert flagged["source_compatible"]["candidates"][1]["config"] == 18
    q1 = StagedGQAPrefill().report(native(((1, 8),), 256))
    assert q1["issue_kind"] is None
    assert q1["source_compatible"]["class"] is None
    assert q1["source_compatible"]["reason"] == "q_len_1_incre_flash_attention"
    roped = StagedGQAPrefill().report(native(((32, 0),), 256, rope=True))
    assert roped["issue_kind"] is None
    assert roped["source_compatible"]["reason"] == "outside_original_abi"
