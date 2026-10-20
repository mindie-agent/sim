"""Finite fusion pipeline invariants.

These tests exercise the emitted finite task/buffer witnesses, not prose.  They
build normalized OpSpec fixtures, execute the actual stage graph on the shared
resource engine, and compare the online-softmax recurrence with an independent
stable full-softmax reference.
"""
from __future__ import annotations

import math
import re
from dataclasses import replace

import pytest

from mindie_sim.a3_cost import CalibrationResult, CostedOp
from mindie_sim.a3_fusion import apply_fusion
from mindie_sim.a3_fusion_placement import place_full_work_fusions
from mindie_sim.a3_ops import IterationWork, OpSpec
from mindie_sim.a3_resources import ResourceEngine, ResourceSpec, add_costed_ops
from mindie_sim.a3_workload import IterationPlan

CAL = CalibrationResult("normalized_oracle", 1, 0, None, False, False, "declared_work")


def op(name, family, shape, deps=(), dtype="bf16", **work):
    return OpSpec(
        name, family, "prefill", 0, 0, ("r0",), tuple(shape), tuple(shape),
        tuple(shape), tuple(shape), deps=tuple(deps), dtype=dtype, **work,
    )


def work_of(*ops):
    return IterationWork(tuple(ops), IterationPlan(()), 1, 1, 1, None, "prefill", "fixture")


def fused_op(result):
    return next(op for op in result.work.ops if op.coverage == "fused")


def metadata(result):
    return dict(fused_op(result).metadata)


def stage_by_name(meta, name):
    return next(stage for stage in meta["fusion_stages"] if stage["name"] == name)


def execute(work):
    capacities = {
        key: ResourceSpec(key, 1.0) for key in
        ("cube", "vector", "hbm", "mte2", "mte3", "launch", "serial")
    }
    engine = ResourceEngine(capacities)
    ids = add_costed_ops(
        engine, [CostedOp(item, (), 0, 0, CAL) for item in work.ops],
        {0: {key: key for key in capacities}}, prefix="fixture", start_ms=0.0,
    )
    return engine.run(), ids


_STAGE_RE = re.compile(r":stage(\d+):")


def actual_peak_live_bytes(meta, run):
    by_index = {}
    for event in run["trace"]:
        match = _STAGE_RE.search(event["task_id"])
        if match:
            by_index[int(match.group(1))] = event
    events = []
    for buffer in meta["buffer_plan"]:
        created = by_index[buffer["created_stage"]]["end_ms"]
        released = by_index[buffer["release_stage"]]["end_ms"]
        events.append((created, 1, buffer["bytes"]))
        events.append((released, -1, buffer["bytes"]))
    # At a shared timestamp, admit all starts before releases; this is
    # conservative rather than assuming an unmodeled intra-instant lifetime.
    events.sort(key=lambda item: (item[0], -item[1]))
    live = peak = 0.0
    for _, sign, size in events:
        live += sign * size
        peak = max(peak, live)
    return peak


def assert_buffer_lifetimes(meta, run):
    """Release follows every consumer and every consumer is ordered to it."""
    by_index = {}
    for event in run["trace"]:
        match = _STAGE_RE.search(event["task_id"])
        if match:
            by_index[int(match.group(1))] = event
    parents = {
        stage["stage_index"]: tuple(stage["deps"])
        for stage in meta["fusion_stages"]
    }

    def ancestors(index):
        seen = set()
        stack = list(parents.get(index, ()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(parents.get(current, ()))
        return seen

    for buffer in meta["buffer_plan"]:
        release = buffer["release_stage"]
        release_end = by_index[release]["end_ms"]
        release_ancestors = ancestors(release)
        for consumer in buffer["consumers"]:
            assert by_index[consumer]["end_ms"] <= release_end + 1e-8
            if consumer != release:
                assert consumer in release_ancestors
    intervals = {}
    for buffer in meta["buffer_plan"]:
        key = (buffer["pool"], buffer["category"], buffer["slot"])
        end = max(
            by_index[index]["end_ms"]
            for index in set(buffer["consumers"]) | {buffer["release_stage"]}
        )
        intervals.setdefault(key, []).append(
            (by_index[buffer["created_stage"]]["start_ms"], end, buffer["id"])
        )
    for values in intervals.values():
        values.sort()
        for prior, current in zip(values, values[1:]):
            assert current[0] + 1e-8 >= prior[1]


def actual_pool_peak(meta, run, pool):
    by_index = {}
    for event in run["trace"]:
        match = _STAGE_RE.search(event["task_id"])
        if match:
            by_index[int(match.group(1))] = event
    ticks = []
    for buffer in meta["buffer_plan"]:
        if buffer["pool"] != pool:
            continue
        start = by_index[buffer["created_stage"]]["start_ms"]
        end = max(by_index[index]["end_ms"] for index in buffer["consumers"])
        ticks.append((start, 1, buffer["bytes"]))
        ticks.append((end, -1, buffer["bytes"]))
    ticks.sort(key=lambda item: (item[0], -item[1]))
    live = peak = 0.0
    for _, sign, amount in ticks:
        live += sign * amount
        peak = max(peak, live)
    return peak


def stage_dependency_closure(meta):
    parents = {
        stage["stage_index"]: tuple(stage["deps"])
        for stage in meta["fusion_stages"]
    }

    def closure(index):
        seen = set()
        stack = list(parents.get(index, ()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(parents.get(current, ()))
        return seen

    return parents, closure


def reference_softmax(scores, values):
    rows = []
    for row in scores:
        finite = [value for value in row if math.isfinite(value)]
        maximum = max(finite) if finite else -math.inf
        weights = [
            math.exp(value - maximum) if math.isfinite(value) else 0.0
            for value in row
        ]
        denominator = sum(weights)
        rows.append([
            sum(weight * value[j] for weight, value in zip(weights, values))
            / denominator
            for j in range(len(values[0]))
        ])
    return rows


def online_softmax(scores, values, tile):
    rows = []
    for row in scores:
        maximum, denominator = -math.inf, 0.0
        accumulator = [0.0] * len(values[0])
        for start in range(0, len(row), tile):
            block = row[start:start + tile]
            finite = [value for value in block if math.isfinite(value)]
            if not finite:
                continue
            new_maximum = max(maximum, max(finite))
            alpha = math.exp(maximum - new_maximum) if math.isfinite(maximum) else 0.0
            weights = [
                math.exp(value - new_maximum) if math.isfinite(value) else 0.0
                for value in block
            ]
            denominator = alpha * denominator + sum(weights)
            accumulator = [
                alpha * accumulator[j]
                + sum(
                    weight * value[j]
                    for weight, value in zip(weights, values[start:start + tile])
                )
                for j in range(len(accumulator))
            ]
            maximum = new_maximum
        rows.append([value / denominator for value in accumulator])
    return rows


def test_online_softmax_recurrence_matches_stable_full_reference():
    scores = [
        [
            math.sin(q * 3.0 + k * 0.7) * 7.0 if k <= q + 5 else -math.inf
            for k in range(13)
        ]
        for q in range(8)
    ]
    values = [[math.cos(k + j * 0.3) for j in range(3)] for k in range(13)]
    expected = reference_softmax(scores, values)
    for tile in (1, 2, 4, 7, 16):
        candidate = online_softmax(scores, values, tile)
        error = max(
            abs(left - right)
            for expected_row, candidate_row in zip(expected, candidate)
            for left, right in zip(expected_row, candidate_row)
        )
        assert error < 1e-12


def test_attention_finite_pipeline_conserves_work_and_has_real_tails():
    q_len, kv_len, heads, d_q, d_v = 67, 65, 3, 16, 12
    qk = op("qk", "attention_qk", (q_len, kv_len, heads, d_q),
            flops=2 * q_len * kv_len * heads * d_q)
    soft = op("soft", "attention_softmax", (q_len, kv_len, heads, d_q),
              ("qk",), vector_ops=5 * q_len * kv_len * heads)
    pv = op("pv", "attention_pv", (q_len, kv_len, heads, d_v),
            ("soft",), flops=2 * q_len * kv_len * heads * d_v)
    result = apply_fusion(work_of(qk, soft, pv), scratch_capacity_bytes=12000)
    assert "attention_qk_softmax_pv" in result.applied_rules
    meta = metadata(result)
    stages = meta["fusion_stages"]
    assert meta["tiling"]["query_tail"] == 3
    assert meta["tiling"]["kv_tail"] == 1
    assert meta["tiling"]["head_tail"] == 1
    assert meta["tiling"]["head_tiles"] == 3

    families = {stage["family"] for stage in stages}
    for family in (
        "attention_qk", "attention_softmax_update", "attention_pv",
        "attention_accumulator", "attention_normalize", "attention_write",
    ):
        assert family in families
    for stage in stages:
        assert stage["extents"]
        assert "tile_shape" in stage
        assert isinstance(stage["buffers"], dict)
        assert set(stage) >= {
            "flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes",
        }

    qk_stages = [stage for stage in stages if stage["family"] == "attention_qk"]
    assert any(stage["extents"]["q"][1] == 3 for stage in qk_stages)
    assert any(stage["extents"]["kv"][1] == 1 for stage in qk_stages)
    assert any(stage["extents"]["head"][1] == 1 for stage in qk_stages)
    assert sum(stage["flops"] for stage in stages) == pytest.approx(
        qk.flops + pv.flops
    )
    assert sum(stage["vector_ops"] for stage in stages) >= soft.vector_ops
    assert sum(stage["hbm_bytes"] for stage in stages) == pytest.approx(
        fused_op(result).hbm_bytes
    )
    assert sum(stage["mte2_bytes"] for stage in stages) == pytest.approx(
        fused_op(result).mte2_bytes
    )
    assert sum(stage["mte3_bytes"] for stage in stages) == pytest.approx(
        fused_op(result).mte3_bytes
    )

    # Exact KV/Q/head slot reuse is a task dependency, not a report-only note.
    parents, closure = stage_dependency_closure(meta)
    loads = [stage for stage in stages if stage["name"].startswith("prefetch.")
             and stage["name"].count(".k") == 1]
    for stage in loads:
        assert stage["stage_index"] in parents
    buffer_plan = meta["buffer_plan"]
    assert buffer_plan
    for buffer in buffer_plan:
        assert buffer["created_stage"] is not None
        assert buffer["release_stage"] is not None
        assert buffer["release_stage"] >= buffer["last_consumer_stage"]
        assert "owner" in buffer and "bytes" in buffer and "dtype" in buffer

    run, _ = execute(result.work)
    actual = actual_peak_live_bytes(meta, run)
    assert actual <= meta["peak_live_bytes"] + 1e-9
    assert meta["peak_live_bytes"] <= 12000


def test_attention_two_slots_overlap_and_compete_on_shared_vector():
    q_len, kv_len, heads, d_q, d_v = 16, 65, 1, 16, 16
    qk = op("qk", "attention_qk", (q_len, kv_len, heads, d_q),
            flops=2 * q_len * kv_len * heads * d_q)
    soft = op("soft", "attention_softmax", (q_len, kv_len, heads, d_q),
              ("qk",), vector_ops=5 * q_len * kv_len * heads)
    pv = op("pv", "attention_pv", (q_len, kv_len, heads, d_v),
            ("soft",), flops=2 * q_len * kv_len * heads * d_v)
    result = apply_fusion(work_of(qk, soft, pv), scratch_capacity_bytes=15000)
    assert "attention_qk_softmax_pv" in result.applied_rules
    meta = metadata(result)
    assert meta["tiling"]["kv_slots"] == 2
    base_run, _ = execute(result.work)
    trace = base_run["trace"]
    qk_events = [event for event in trace if ":qk." in event["owner"]]
    soft_events = [event for event in trace
                   if ":softmax" in event["owner"]]
    overlaps = [
        (cube, vector)
        for cube in qk_events for vector in soft_events
        if cube["start_ms"] < vector["end_ms"]
        and vector["start_ms"] < cube["end_ms"]
    ]
    assert overlaps
    final_name = next(
        stage["name"] for stage in meta["fusion_stages"]
        if stage["family"] == "attention_write"
        and not any(
            other["family"] == "attention_write"
            and other["stage_index"] > stage["stage_index"]
            for other in meta["fusion_stages"]
        )
    )
    base_final = max(
        event["end_ms"] for event in trace if event["owner"].endswith(final_name)
    )

    def with_competitor(competitor_ops, release_ms):
        capacities = {
            key: ResourceSpec(key, 1.0) for key in
            ("cube", "vector", "hbm", "mte2", "mte3", "launch", "serial")
        }
        engine = ResourceEngine(capacities)
        add_costed_ops(
            engine, [CostedOp(item, (), 0, 0, CAL) for item in result.work.ops],
            {0: {key: key for key in capacities}}, prefix="fixture", start_ms=0.0,
        )
        engine.add_task("competitor", {"vector": float(competitor_ops)},
                        release_ms=release_ms)
        return engine.run()

    soft_start = min(event["start_ms"] for event in soft_events)
    hidden = with_competitor(1_000_000.0, base_final + 100.0)
    hidden_final = max(
        event["end_ms"] for event in hidden["trace"]
        if event["owner"].endswith(final_name)
    )
    assert hidden_final == pytest.approx(base_final)
    exposed = with_competitor(1_000_000.0, soft_start)
    exposed_final = max(
        event["end_ms"] for event in exposed["trace"]
        if event["owner"].endswith(final_name)
    )
    assert exposed_final > base_final

    # A nonuniform final KV tail carries smaller score/PV extents than a full
    # tile, and its accumulator work is correspondingly smaller.
    accum = [stage for stage in meta["fusion_stages"]
             if stage["family"] == "attention_accumulator"]
    full = max(accum, key=lambda stage: stage["extents"]["kv"][1])
    tail = min(accum, key=lambda stage: stage["extents"]["kv"][1])
    assert tail["extents"]["kv"][1] < full["extents"]["kv"][1]
    assert (tail["extents"]["kv"][1] * tail["extents"]["q"][1]
            < full["extents"]["kv"][1] * full["extents"]["q"][1])


def test_exact_norm_row_tile_bf16_full_k_and_slot_reuse():
    m, n, k = 1024, 768, 512
    x_bytes, w_bytes, y_bytes = 2 * m * k, 2 * k * n, 2 * m * n
    norm = op("norm", "rmsnorm", (m, k), vector_ops=5 * m * k,
              hbm_bytes=2 * x_bytes, mte2_bytes=x_bytes, mte3_bytes=x_bytes)
    gemm = op("gemm", "qkv_projection", (m, n, k), ("norm",),
              flops=2 * m * n * k,
              hbm_bytes=x_bytes + w_bytes + y_bytes,
              mte2_bytes=x_bytes + w_bytes, mte3_bytes=y_bytes)
    result = apply_fusion(work_of(norm, gemm), scratch_capacity_bytes=65536)
    assert "rmsnorm_gemm_epilogue" in result.applied_rules
    fused = fused_op(result)
    meta = dict(fused.metadata)
    tiling = meta["tiling"]
    assert (tiling["row_tile"], tiling["n_tile"], tiling["k_tile"]) == (16, 64, 64)
    assert meta["peak_live_bytes"] == 57472
    assert meta["peak_live_bytes"] <= 65536
    assert tiling["row_slots"] == 2
    assert tiling["weight_slots"] == 2
    assert tiling["output_slots"] == 2

    kinds = [stage["kind"] for stage in meta["fusion_stages"]]
    for kind in (
        "rms_reduce_full_k", "rms_scale_cast_bf16", "weight_load",
        "gemm_k_accumulate", "output_cast_bf16", "output_write",
    ):
        assert kind in kinds
    reduce_stages = [
        stage for stage in meta["fusion_stages"]
        if stage["kind"] == "rms_reduce_full_k"
    ]
    assert len(reduce_stages) == tiling["row_tiles"]
    assert all(stage["extents"]["k"] == [0, k] for stage in reduce_stages)
    assert all(stage["extents"]["full_k"] for stage in reduce_stages)
    normalized_rows = [
        buffer for buffer in meta["buffer_plan"]
        if buffer["category"] == "normalized_row"
    ]
    assert normalized_rows and all(buffer["dtype"] == "bf16"
                                  for buffer in normalized_rows)

    assert sum(stage["flops"] for stage in meta["fusion_stages"]) == pytest.approx(
        gemm.flops
    )
    assert fused.vector_ops == pytest.approx(norm.vector_ops + m * n)
    compact = any(
        stage["extents"].get("compact_inner_loop") is True
        for stage in meta["fusion_stages"]
    )
    weight_loads = [
        stage for stage in meta["fusion_stages"]
        if stage["kind"] == "weight_load"
    ]
    expected_weight_stages = (
        tiling["row_tiles"] * tiling["n_tiles"]
        if compact else
        tiling["row_tiles"] * tiling["n_tiles"] * tiling["k_tiles"]
    )
    assert len(weight_loads) == expected_weight_stages
    assert sum(stage["mte2_bytes"] for stage in weight_loads) == pytest.approx(
        tiling["row_tiles"] * w_bytes
    )
    assert sum(stage["mte2_bytes"] for stage in meta["fusion_stages"]) == pytest.approx(
        fused.mte2_bytes
    )
    assert fused.mte2_bytes == pytest.approx(x_bytes + tiling["row_tiles"] * w_bytes)
    assert sum(stage["mte3_bytes"] for stage in meta["fusion_stages"]) == pytest.approx(
        fused.mte3_bytes
    )
    assert fused.mte3_bytes == pytest.approx(y_bytes)

    # Row slot 0 is reused by row 32 only after the explicit join of every
    # row-0 GEMM consumer.  Row 1 uses the other slot, so its norm does not
    # wait for row 0's GEMM.
    row0_last = stage_by_name(
        meta, "gemm.r0.n704.k0" if compact else "gemm.r0.n704.k448"
    )
    row_release = stage_by_name(meta, "release.row.r0")
    row1_load = stage_by_name(meta, "norm.load.r16")
    row32_load = stage_by_name(meta, "norm.load.r32")
    row0_scale = stage_by_name(meta, "norm.scale.r0")
    assert row0_scale["stage_index"] in row1_load["deps"]
    assert row0_last["stage_index"] not in row1_load["deps"]
    assert row0_last["stage_index"] in row_release["deps"]
    assert row_release["stage_index"] in row32_load["deps"]
    # A weight slot is reused only after the GEMM consuming it.
    first_weight = stage_by_name(meta, "weight.load.r0.n0.k0")
    reuse_weight = stage_by_name(meta, "weight.load.r0.n128.k0")
    assert first_weight["stage_index"] not in reuse_weight["deps"]
    by_index = {stage["stage_index"]: stage for stage in meta["fusion_stages"]}
    assert any(
        by_index[index]["name"].startswith("gemm.r0.")
        for index in reuse_weight["deps"]
    )
    # Output slot 0 is reused only after its prior write.
    assert stage_by_name(
        meta, "output.write.r0.n0"
    )["stage_index"] in stage_by_name(meta, "output.cast.r0.n128")["deps"]

    run, _ = execute(result.work)
    actual = actual_peak_live_bytes(meta, run)
    assert actual <= meta["peak_live_bytes"] + 1e-9
    norm_scale = next(
        event for event in run["trace"] if "norm.scale.r16" in event["owner"]
    )
    row0_gemms = [
        event for event in run["trace"]
        if event["owner"].split("fused:gemm:")[-1].startswith("gemm.r0.")
    ]
    assert any(
        event["start_ms"] < norm_scale["end_ms"]
        and norm_scale["start_ms"] < event["end_ms"]
        for event in row0_gemms
    )


def test_norm_two_pass_counts_reread_when_one_full_row_does_not_fit():
    m, n, k = 8, 1, 512
    x_bytes, w_bytes, y_bytes = 2 * m * k, 2 * k * n, 2 * m * n
    norm = op("norm", "rmsnorm", (m, k), vector_ops=5 * m * k,
              hbm_bytes=2 * x_bytes, mte2_bytes=x_bytes, mte3_bytes=x_bytes)
    gemm = op("gemm", "qkv_projection", (m, n, k), ("norm",),
              flops=2 * m * n * k,
              hbm_bytes=x_bytes + w_bytes + y_bytes,
              mte2_bytes=x_bytes + w_bytes, mte3_bytes=y_bytes)
    result = apply_fusion(work_of(norm, gemm), scratch_capacity_bytes=1000)
    assert "rmsnorm_gemm_epilogue" in result.applied_rules
    fused = fused_op(result)
    meta = dict(fused.metadata)
    assert meta["tiling"]["mode"] == "two_pass"
    assert meta["two_pass"] is True
    assert meta["x_passes"] == 2
    assert meta["extra_x_read_bytes"] == pytest.approx(x_bytes)
    assert any(stage["kind"] == "rms_second_pass_load"
               for stage in meta["fusion_stages"])
    assert any(stage["kind"] == "rms_scale_cast_partial"
               for stage in meta["fusion_stages"])
    assert sum(stage["hbm_bytes"] for stage in meta["fusion_stages"]) == pytest.approx(
        fused.hbm_bytes
    )
    assert fused.hbm_bytes == pytest.approx(2 * x_bytes + w_bytes + y_bytes)
    assert sum(stage["flops"] for stage in meta["fusion_stages"]) == pytest.approx(
        gemm.flops
    )
    assert meta["peak_live_bytes"] <= 1000


def test_attention_removes_only_resident_score_and_kv_reread_bytes():
    q_len, kv_len, heads, d_q, d_v = 8, 4, 2, 8, 8
    q_bytes = q_len * heads * d_q * 2
    kv_bytes = kv_len * heads * d_q * 2
    score_bytes = q_len * kv_len * heads * 2
    out_bytes = q_len * heads * d_v * 2
    qk = op("qk", "attention_qk", (q_len, kv_len, heads, d_q),
            flops=2 * q_len * kv_len * heads * d_q,
            hbm_bytes=q_bytes + kv_bytes + score_bytes,
            mte2_bytes=q_bytes + kv_bytes, mte3_bytes=score_bytes)
    soft = op("soft", "attention_softmax", (q_len, kv_len, heads, d_q),
              ("qk",), vector_ops=5 * q_len * kv_len * heads,
              hbm_bytes=4 * score_bytes, mte2_bytes=2 * score_bytes,
              mte3_bytes=2 * score_bytes)
    pv = op("pv", "attention_pv", (q_len, kv_len, heads, d_v),
            ("soft",), flops=2 * q_len * kv_len * heads * d_v,
            hbm_bytes=kv_bytes + score_bytes + out_bytes,
            mte2_bytes=kv_bytes + score_bytes, mte3_bytes=out_bytes)
    result = apply_fusion(work_of(qk, soft, pv), scratch_capacity_bytes=65536)
    assert "attention_qk_softmax_pv" in result.applied_rules
    fused = fused_op(result)
    # Q/K/V are loaded once.  V is not implied by qk.mte2 and must remain:
    # q_bytes + K bytes + V bytes, then only the final output is written.
    assert fused.mte2_bytes == pytest.approx(q_bytes + kv_bytes + kv_bytes)
    assert fused.mte3_bytes == pytest.approx(out_bytes)
    assert fused.hbm_bytes == pytest.approx(
        q_bytes + kv_bytes + kv_bytes + out_bytes
    )
    assert result.changes["mte2_bytes"] < 0
    assert result.changes["hbm_bytes"] < 0
    assert sum(stage["flops"] for stage in dict(fused.metadata)["fusion_stages"]) \
        == pytest.approx(qk.flops + pv.flops)


def test_public_long_context_attention_is_bounded_not_rejected():
    q_len, kv_len, heads, d_q, d_v = 1, 32769, 8, 64, 64
    qk = op("qk", "attention_qk", (q_len, kv_len, heads, d_q),
            flops=2 * q_len * kv_len * heads * d_q)
    soft = op("soft", "attention_softmax", (q_len, kv_len, heads, d_q),
              ("qk",), vector_ops=5 * q_len * kv_len * heads)
    pv = op("pv", "attention_pv", (q_len, kv_len, heads, d_v),
            ("soft",), flops=2 * q_len * kv_len * heads * d_v)
    result = apply_fusion(work_of(qk, soft, pv), scratch_capacity_bytes=262144)
    assert "attention_qk_softmax_pv" in result.applied_rules
    meta = metadata(result)
    assert meta["tiling"]["kv_tiles"] > 1
    assert meta["tiling"]["kv_tail"] == 1
    assert meta["tiling"]["kv_slots"] == 2
    assert meta["required_by_pool"]["l1"] + meta["required_by_pool"]["ub"] \
        + meta["required_by_pool"]["l0c"] <= 262144
    assert len(meta["fusion_stages"]) < 10000


def test_declared_pool_capacities_are_consumed_and_scope_is_explicit():
    q_len, kv_len, heads, d_q, d_v = 64, 64, 2, 16, 16
    qk = op("qk", "attention_qk", (q_len, kv_len, heads, d_q),
            flops=2 * q_len * kv_len * heads * d_q)
    soft = op("soft", "attention_softmax", (q_len, kv_len, heads, d_q),
              ("qk",), vector_ops=5 * q_len * kv_len * heads)
    pv = op("pv", "attention_pv", (q_len, kv_len, heads, d_v),
            ("soft",), flops=2 * q_len * kv_len * heads * d_v)
    pools = {"l1": 10000, "ub": 100000, "l0a": 100000,
             "l0b": 100000, "l0c": 100000}
    result = apply_fusion(
        work_of(qk, soft, pv), scratch_capacity_bytes=10_000_000,
        pool_capacities=pools,
    )
    assert "attention_qk_softmax_pv" in result.applied_rules
    meta = metadata(result)
    for pool, amount in meta["required_by_pool"].items():
        assert amount <= pools[pool]
    assert result.witness["memory_scope"] == "declared_pools"
    assert result.witness["scratch_pool"]["pool_coverage_complete"] is True
    assert meta["physical_feasible"] is True
    rejected = apply_fusion(
        work_of(qk, soft, pv), scratch_capacity_bytes=10_000_000,
        pool_capacities={"l1": 1, "ub": 1, "l0a": 1, "l0b": 1, "l0c": 1},
    )
    assert "attention_qk_softmax_pv" not in rejected.applied_rules
    assert rejected.considered_rules[-1]["reason"] == "live_buffer_capacity_exceeded"


def test_attention_conserves_physical_qkv_with_unequal_dq_dv():
    q_len, kv_len, heads, d_q, d_v = 3, 13, 1, 4, 7
    q_bytes = 2 * q_len * heads * d_q
    k_bytes = 2 * kv_len * heads * d_q
    v_bytes = 2 * kv_len * heads * d_v
    out_bytes = 2 * q_len * heads * d_v
    scores = 4 * q_len * kv_len * heads
    qk = op("qk", "attention_qk", (q_len, kv_len, heads, d_q),
            flops=2 * q_len * kv_len * heads * d_q,
            hbm_bytes=q_bytes + k_bytes + scores,
            mte2_bytes=q_bytes + k_bytes, mte3_bytes=scores)
    soft = op("soft", "attention_softmax", (q_len, kv_len, heads, d_q),
              ("qk",), dtype="fp32", vector_ops=5 * q_len * kv_len * heads,
              hbm_bytes=2 * scores, mte2_bytes=scores, mte3_bytes=scores)
    pv = op("pv", "attention_pv", (q_len, kv_len, heads, d_v),
            ("soft",), flops=2 * q_len * kv_len * heads * d_v,
            hbm_bytes=scores + v_bytes + out_bytes,
            mte2_bytes=scores + v_bytes, mte3_bytes=out_bytes)
    pools = {"l1": 524288, "ub": 196608, "l0a": 65536,
             "l0b": 65536, "l0c": 131072}
    result = apply_fusion(work_of(qk, soft, pv), pool_capacities=pools)
    assert "attention_qk_softmax_pv" in result.applied_rules
    fused = fused_op(result)
    meta = dict(fused.metadata)
    assert fused.mte2_bytes == pytest.approx(q_bytes + k_bytes + v_bytes)
    assert fused.hbm_bytes == pytest.approx(
        q_bytes + k_bytes + v_bytes + out_bytes
    )
    assert fused.mte3_bytes == pytest.approx(out_bytes)
    assert sum(stage["mte2_bytes"] for stage in meta["fusion_stages"])         == pytest.approx(fused.mte2_bytes)
    q_load = stage_by_name(meta, "load.q.q0.h0")
    kv_load = stage_by_name(meta, "prefetch.kv.q0.h0.k0")
    assert q_load["mte2_bytes"] == pytest.approx(q_bytes)
    assert kv_load["mte2_bytes"] == pytest.approx(k_bytes + v_bytes)
    created = {item["category"]: item for item in kv_load["buffers"]["create"]}
    assert created["kv_key"]["bytes"] == pytest.approx(k_bytes)
    assert created["kv_value"]["bytes"] == pytest.approx(v_bytes)
    v_buffer = next(buffer for buffer in meta["buffer_plan"]
                    if buffer["category"] == "kv_value")
    assert v_buffer["dtype"] == "bf16"
    assert any(stage["kind"] == "accumulator_update"
               and stage["vector_ops"] > 0 for stage in meta["fusion_stages"])
    assert all(
        not stage["buffers"].get("write")
        or any(stage[key] > 0 for key in
               ("flops", "vector_ops", "mte2_bytes", "mte3_bytes", "hbm_bytes"))
        for stage in meta["fusion_stages"]
    )
    run, _ = execute(result.work)
    assert actual_peak_live_bytes(meta, run) <= meta["peak_live_bytes"]
    assert_buffer_lifetimes(meta, run)


def test_two_pass_norm_release_joins_all_parallel_gemm_consumers():
    m, n, k = 1, 65, 512
    x_bytes, w_bytes, y_bytes = 2 * m * k, 2 * k * n, 2 * m * n
    norm = op("norm", "rmsnorm", (m, k), vector_ops=5 * m * k,
              hbm_bytes=2 * x_bytes, mte2_bytes=x_bytes, mte3_bytes=x_bytes)
    gemm = op("gemm", "qkv_projection", (m, n, k), ("norm",),
              flops=2 * m * n * k,
              hbm_bytes=x_bytes + w_bytes + y_bytes,
              mte2_bytes=x_bytes + w_bytes, mte3_bytes=y_bytes)
    result = apply_fusion(
        work_of(norm, gemm),
        pool_capacities={"l1": 1000, "ub": 128, "l0c": 1024},
    )
    assert "rmsnorm_gemm_epilogue" in result.applied_rules
    meta = metadata(result)
    assert meta["tiling"]["mode"] == "two_pass"
    run, _ = execute(result.work)
    assert_buffer_lifetimes(meta, run)
    release = stage_by_name(meta, "release.x.r0.g0.k16")
    consumers = [
        stage["stage_index"] for stage in meta["fusion_stages"]
        if stage["name"].startswith("gemm.r0.g0.n") and
        stage["name"].endswith(".k16")
    ]
    assert consumers and set(consumers) <= set(release["deps"])
    tail_release = stage_by_name(meta, "release.x.r0.g0.k496")
    tail_consumers = [
        stage["stage_index"] for stage in meta["fusion_stages"]
        if stage["name"].startswith("gemm.r0.g0.n") and
        stage["name"].endswith(".k496")
    ]
    assert set(tail_consumers) <= set(tail_release["deps"])


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_semantic_softmax_work_and_paired_dtype_recipe(dtype):
    q_len, kv_len, heads, d_q, d_v = 1, 2, 1, 128, 128
    q_bytes = 2 * q_len * heads * d_q
    k_bytes = 2 * kv_len * heads * d_q
    v_bytes = 2 * kv_len * heads * d_v
    out_bytes = 2 * q_len * heads * d_v
    scores = 4 * q_len * kv_len * heads
    qk = op("qk", "attention_qk", (q_len, kv_len, heads, d_q),
            dtype=dtype, flops=2 * q_len * kv_len * heads * d_q,
            hbm_bytes=q_bytes + k_bytes + scores,
            mte2_bytes=q_bytes + k_bytes, mte3_bytes=scores)
    soft = op("soft", "attention_softmax", (q_len, kv_len, heads, d_q),
              ("qk",), dtype="fp32", vector_ops=5 * q_len * kv_len * heads,
              hbm_bytes=2 * scores, mte2_bytes=scores, mte3_bytes=scores)
    pv = op("pv", "attention_pv", (q_len, kv_len, heads, d_v),
            ("soft",), dtype=dtype, flops=2 * q_len * kv_len * heads * d_v,
            hbm_bytes=scores + v_bytes + out_bytes,
            mte2_bytes=scores + v_bytes, mte3_bytes=out_bytes)
    pools = {"l1": 524288, "ub": 196608, "l0a": 65536,
             "l0b": 65536, "l0c": 131072}
    result = apply_fusion(
        work_of(qk, soft, pv), pool_capacities=pools,
        core_count=16, per_core_rates={"cube": 1.0, "vector": 1.0},
        core_base=3,
    )
    assert "attention_qk_softmax_pv" in result.applied_rules
    meta = metadata(result)
    assert meta["physical_feasible"] is True
    soft_stages = [stage for stage in meta["fusion_stages"]
                   if stage["kind"] == "softmax_update"]
    norm_stages = [stage for stage in meta["fusion_stages"]
                   if stage["kind"] == "normalize"]
    assert sum(stage["vector_ops"] for stage in soft_stages) > 0
    assert sum(stage["vector_ops"] for stage in norm_stages) >= (
        q_len * heads * d_v
    )
    pv_stage = next(stage for stage in meta["fusion_stages"]
                    if stage["kind"] == "pv")
    operand_dtypes = {
        item["dtype"] for item in pv_stage["buffers"]["read"]
        if item["category"] in {
            "pv_l0a", "pv_l0b", "kv_value", "probability_tile",
            "probability_operand",
        }
    }
    assert operand_dtypes == {dtype}
    output_cast = next(stage for stage in meta["fusion_stages"]
                       if "output_to_" in stage["kind"])
    assert output_cast["extents"]["dst_dtype"] == dtype
    assert all(
        stage["core_assignment"]["core_ids"] in ((3,), ())
        for stage in meta["fusion_stages"]
    )
    assert all(
        stage["execution_width"]["partition"] == "single_core_tile"
        for stage in meta["fusion_stages"]
    )


def test_physical_l0_path_and_zero_l0_scope():
    q_len, kv_len, heads, d_q, d_v = 3, 13, 1, 4, 7
    q_bytes = 2 * q_len * heads * d_q
    k_bytes = 2 * kv_len * heads * d_q
    v_bytes = 2 * kv_len * heads * d_v
    out_bytes = 2 * q_len * heads * d_v
    scores = 4 * q_len * kv_len * heads
    qk = op("qk", "attention_qk", (q_len, kv_len, heads, d_q),
            flops=2 * q_len * kv_len * heads * d_q,
            hbm_bytes=q_bytes + k_bytes + scores,
            mte2_bytes=q_bytes + k_bytes, mte3_bytes=scores)
    soft = op("soft", "attention_softmax", (q_len, kv_len, heads, d_q),
              ("qk",), dtype="fp32", vector_ops=5 * q_len * kv_len * heads,
              hbm_bytes=2 * scores, mte2_bytes=scores, mte3_bytes=scores)
    pv = op("pv", "attention_pv", (q_len, kv_len, heads, d_v),
            ("soft",), flops=2 * q_len * kv_len * heads * d_v,
            hbm_bytes=scores + v_bytes + out_bytes,
            mte2_bytes=scores + v_bytes, mte3_bytes=out_bytes)
    pools = {"l1": 524288, "ub": 196608, "l0a": 65536,
             "l0b": 65536, "l0c": 131072}
    result = apply_fusion(work_of(qk, soft, pv), pool_capacities=pools)
    assert "attention_qk_softmax_pv" in result.applied_rules
    fused = fused_op(result)
    meta = dict(fused.metadata)
    assert meta["physical_feasible"] is True
    assert meta["physical_path"]
    assert fused.mte2_bytes == pytest.approx(q_bytes + k_bytes + v_bytes)
    l0_buffers = [buffer for buffer in meta["buffer_plan"]
                  if buffer["pool"] in ("l0a", "l0b")]
    assert l0_buffers
    assert max(buffer["bytes"] for buffer in l0_buffers) <= 65536
    assert any(buffer["category"] == "probability_operand"
               for buffer in meta["buffer_plan"])
    assert any(buffer["category"] == "output_bf16"
               for buffer in meta["buffer_plan"])
    assert any(stage["kind"] == "fp32_prob_to_bf16"
               for stage in meta["fusion_stages"])
    assert any(stage["kind"] == "fp32_output_to_bf16"
               for stage in meta["fusion_stages"])
    pv_stage = next(stage for stage in meta["fusion_stages"]
                    if stage["kind"] == "pv")
    assert {item["dtype"] for item in pv_stage["buffers"]["read"]} <= {"bf16"}
    assert all(
        not stage["buffers"].get("write")
        or any(stage[key] > 0 for key in
               ("flops", "vector_ops", "mte2_bytes", "mte3_bytes", "hbm_bytes"))
        for stage in meta["fusion_stages"]
    )

    logical = apply_fusion(
        work_of(qk, soft, pv),
        pool_capacities={**pools, "l0a": 0, "l0b": 0},
    )
    assert "attention_qk_softmax_pv" in logical.applied_rules
    logical_meta = metadata(logical)
    assert logical_meta["physical_feasible"] is False
    assert logical.witness["memory_scope"] != "declared_pools"
    required = apply_fusion(
        work_of(qk, soft, pv),
        pool_capacities={**pools, "l0a": 0, "l0b": 0},
        require_physical=True,
    )
    assert "attention_qk_softmax_pv" not in required.applied_rules
    assert required.considered_rules[-1]["reason"] == (
        "physical_pool_path_unavailable"
    )


def test_physical_norm_splits_l0b_uses_ub_and_copies_l0c():
    m, n, k = 1024, 768, 512
    x_bytes, w_bytes, y_bytes = 2 * m * k, 2 * k * n, 2 * m * n
    norm = op("norm", "rmsnorm", (m, k), vector_ops=5 * m * k,
              hbm_bytes=2 * x_bytes, mte2_bytes=x_bytes, mte3_bytes=x_bytes)
    gemm = op("gemm", "qkv_projection", (m, n, k), ("norm",),
              flops=2 * m * n * k,
              hbm_bytes=x_bytes + w_bytes + y_bytes,
              mte2_bytes=x_bytes + w_bytes, mte3_bytes=y_bytes)
    pools = {"l1": 524288, "ub": 196608, "l0a": 65536,
             "l0b": 65536, "l0c": 131072}
    result = apply_fusion(work_of(norm, gemm), pool_capacities=pools)
    assert "rmsnorm_gemm_epilogue" in result.applied_rules
    fused = fused_op(result)
    meta = dict(fused.metadata)
    assert meta["physical_feasible"] is True
    assert max(buffer["bytes"] for buffer in meta["buffer_plan"]
               if buffer["pool"] == "l0b") <= 65536
    assert any(stage["kind"] == "ub_to_l1_handoff"
               for stage in meta["fusion_stages"])
    assert any(stage["kind"] == "l0c_to_ub_output"
               for stage in meta["fusion_stages"])
    assert not any(
        stage["vector_ops"] > 0
        and any(item["pool"] == "l1" for item in stage["buffers"].get("read", ()))
        for stage in meta["fusion_stages"]
    )
    assert sum(stage["flops"] for stage in meta["fusion_stages"]) == pytest.approx(
        gemm.flops
    )


def _captured_attention_branch(tag, parents):
    q_len, kv_len, heads, d_q, d_v = 8, 1, 4, 512, 512
    q_bytes = 2 * q_len * heads * d_q
    k_bytes = 2 * kv_len * heads * d_q
    v_bytes = 2 * kv_len * heads * d_v
    out_bytes = 2 * q_len * heads * d_v
    scores = 4 * q_len * kv_len * heads
    qk = op(f"{tag}.qk", "attention_qk", (q_len, kv_len, heads, d_q),
            parents, flops=2 * q_len * kv_len * heads * d_q,
            mte2_bytes=q_bytes + k_bytes,
            mte3_bytes=scores, hbm_bytes=q_bytes + k_bytes + scores)
    soft = op(f"{tag}.soft", "attention_softmax", (q_len, kv_len, heads, d_q),
              (qk.op_id,), dtype="fp32", vector_ops=5 * q_len * kv_len * heads,
              mte2_bytes=scores, mte3_bytes=scores, hbm_bytes=2 * scores)
    pv = op(f"{tag}.pv", "attention_pv", (q_len, kv_len, heads, d_v),
            (soft.op_id,), flops=2 * q_len * kv_len * heads * d_v,
            mte2_bytes=scores + v_bytes, mte3_bytes=out_bytes,
            hbm_bytes=scores + v_bytes + out_bytes)
    return [qk, soft, pv]


def test_full_work_placement_disjoint_or_serialized_by_fixed_core():
    parents = [
        op("q_b", "projection", (8, 4, 512)),
        op("swa_store", "cache", (1, 4, 512)),
    ]
    ops = parents + _captured_attention_branch("early", ("q_b", "swa_store")) \
        + _captured_attention_branch("late", ("q_b", "swa_store"))
    pools = {"l1": 524288, "ub": 196608, "l0a": 65536,
             "l0b": 65536, "l0c": 131072}

    auto = apply_fusion(
        work_of(*ops), pool_capacities=pools, core_count=8,
        worker_count=1, core_base=None,
    )
    auto_fused = [item for item in auto.work.ops if item.coverage == "fused"]
    assert len(auto_fused) == 2
    auto_cores = {}
    for item in auto_fused:
        meta = dict(item.metadata)
        assert meta["physical_feasible"] is True
        auto_cores[item.op_id] = meta["core_contract"]["worker_cores"]["0"]
    assert sorted(auto_cores.values()) == [0, 1]
    assert auto.witness["full_work_placement"]["added_serialization_edges"] == []
    early_id, late_id = sorted(auto_cores)
    late = next(item for item in auto_fused if item.op_id == late_id)
    assert early_id not in late.deps
    auto_run, _ = execute(auto.work)
    auto_events = {item.op_id: [] for item in auto_fused}
    for event in auto_run["trace"]:
        for item in auto_fused:
            if f":{item.op_id}:stage" in event["task_id"]:
                auto_events[item.op_id].append(event)
    assert min(event["start_ms"] for event in auto_events[late_id]) < max(
        event["end_ms"] for event in auto_events[early_id]
    )

    fixed = apply_fusion(
        work_of(*ops), pool_capacities=pools, core_count=8,
        worker_count=1, core_base=0,
    )
    fixed_fused = [item for item in fixed.work.ops if item.coverage == "fused"]
    assert len(fixed_fused) == 2
    assert {dict(item.metadata)["core_contract"]["worker_cores"]["0"]
            for item in fixed_fused} == {0}
    placement = fixed.witness["full_work_placement"]
    assert placement["mode"] == "fixed_core_base"
    assert len(placement["added_serialization_edges"]) == 1
    edge = placement["added_serialization_edges"][0]
    driven = next(item for item in fixed_fused if item.op_id == edge["to"])
    assert edge["from"] in driven.deps
    assert {"from": edge["from"], "to": edge["to"]} in fixed.witness["edges"]

    run, _ = execute(fixed.work)
    by_prefix = {}
    for event in run["trace"]:
        for item in fixed_fused:
            if f":{item.op_id}:stage" in event["task_id"]:
                by_prefix.setdefault(item.op_id, []).append(event)
    first = by_prefix[edge["from"]]
    second = by_prefix[edge["to"]]
    assert max(event["end_ms"] for event in first) <= (
        min(event["start_ms"] for event in second) + 1e-9
    )


def test_tight_ub_attention_uses_pv_partial_in_admission():
    q_len, kv_len, heads, d_q, d_v = 8, 1, 4, 512, 512
    q_bytes = 2 * q_len * heads * d_q
    k_bytes = 2 * kv_len * heads * d_q
    v_bytes = 2 * kv_len * heads * d_v
    out_bytes = 2 * q_len * heads * d_v
    scores = 4 * q_len * kv_len * heads
    qk = op("qk", "attention_qk", (q_len, kv_len, heads, d_q),
            flops=2 * q_len * kv_len * heads * d_q,
            mte2_bytes=q_bytes + k_bytes,
            mte3_bytes=scores, hbm_bytes=q_bytes + k_bytes + scores)
    soft = op("soft", "attention_softmax", (q_len, kv_len, heads, d_q),
              ("qk",), dtype="fp32", vector_ops=5 * q_len * kv_len * heads,
              mte2_bytes=scores, mte3_bytes=scores, hbm_bytes=2 * scores)
    pv = op("pv", "attention_pv", (q_len, kv_len, heads, d_v),
            ("soft",), flops=2 * q_len * kv_len * heads * d_v,
            mte2_bytes=scores + v_bytes, mte3_bytes=out_bytes,
            hbm_bytes=scores + v_bytes + out_bytes)
    pools = {"l1": 524288, "ub": 120000, "l0a": 65536,
             "l0b": 65536, "l0c": 131072}
    result = apply_fusion(
        work_of(qk, soft, pv), pool_capacities=pools,
        core_count=16, require_physical=True,
    )
    assert "attention_qk_softmax_pv" in result.applied_rules
    meta = metadata(result)
    assert meta["physical_feasible"] is True
    assert meta["required_by_pool"]["ub"] <= 120000
    assert meta["tiling"]["head_tile"] == 2
    assert meta["tiling"]["group_slots"] == 1
    run, _ = execute(result.work)
    assert actual_pool_peak(meta, run, "ub") <= 120000
    assert actual_pool_peak(meta, run, "l0a") <= 65536
    assert actual_pool_peak(meta, run, "l0b") <= 65536
    assert actual_pool_peak(meta, run, "l0c") <= 131072


def test_multiworker_norm_has_private_buffers_and_fixed_workers():
    m, n, k = 256, 512, 512
    x_bytes, w_bytes, y_bytes = 2 * m * k, 2 * k * n, 2 * m * n
    norm = op("norm", "rmsnorm", (m, k), vector_ops=5 * m * k,
              hbm_bytes=2 * x_bytes, mte2_bytes=x_bytes, mte3_bytes=x_bytes)
    gemm = op("gemm", "qkv_projection", (m, n, k), ("norm",),
              flops=2 * m * n * k,
              hbm_bytes=x_bytes + w_bytes + y_bytes,
              mte2_bytes=x_bytes + w_bytes, mte3_bytes=y_bytes)
    pools = {"l1": 524288, "ub": 196608, "l0a": 65536,
             "l0b": 65536, "l0c": 131072}
    result = apply_fusion(
        work_of(norm, gemm), pool_capacities=pools,
        core_count=4, per_core_rates={"cube": 1.0, "vector": 1.0},
        worker_count=4,
    )
    assert "rmsnorm_gemm_epilogue" in result.applied_rules
    fused = fused_op(result)
    meta = dict(fused.metadata)
    assert meta["worker_count"] == 4
    assert meta["worker_partition"] == "contiguous_complete_rows"
    assert fused.flops == pytest.approx(gemm.flops)
    contract = meta["core_contract"]
    assert contract["workers_with_work"] == 4
    assert len(set(contract["worker_cores"].values())) == 4
    buffer_ids = [buffer["id"] for buffer in meta["buffer_plan"]]
    assert len(buffer_ids) == len(set(buffer_ids))
    for buffer in meta["buffer_plan"]:
        assert buffer["id"].startswith(f"w{buffer['worker']}.")
    cross_worker = []
    stages = meta["fusion_stages"]
    for buffer in meta["buffer_plan"]:
        uses = {buffer["created_stage"], *buffer["consumers"]}
        workers = {stages[index].get("worker_id", 0) for index in uses
                   if stages[index].get("core_assignment", {}).get("count", 0)}
        if len(workers) > 1:
            cross_worker.append(buffer["id"])
    assert not cross_worker
    for stage in stages:
        if stage["core_assignment"]["count"] > 0:
            worker = stage["worker_id"]
            assert stage["core_assignment"]["core_ids"] == (
                contract["worker_cores"][str(worker)],
            )
    for pool, amount in meta["required_by_pool"].items():
        assert amount <= pools[pool]


def test_multiworker_attention_head_partitions_are_private_and_conserved():
    q_len, kv_len, heads, d_q, d_v = 4, 32, 8, 64, 64
    q_bytes = 2 * q_len * heads * d_q
    k_bytes = 2 * kv_len * heads * d_q
    v_bytes = 2 * kv_len * heads * d_v
    out_bytes = 2 * q_len * heads * d_v
    scores = 4 * q_len * kv_len * heads
    qk = op("qk", "attention_qk", (q_len, kv_len, heads, d_q),
            flops=2 * q_len * kv_len * heads * d_q,
            hbm_bytes=q_bytes + k_bytes + scores,
            mte2_bytes=q_bytes + k_bytes, mte3_bytes=scores)
    soft = op("soft", "attention_softmax", (q_len, kv_len, heads, d_q),
              ("qk",), dtype="fp32", vector_ops=5 * q_len * kv_len * heads,
              hbm_bytes=2 * scores, mte2_bytes=scores, mte3_bytes=scores)
    pv = op("pv", "attention_pv", (q_len, kv_len, heads, d_v),
            ("soft",), flops=2 * q_len * kv_len * heads * d_v,
            hbm_bytes=scores + v_bytes + out_bytes,
            mte2_bytes=scores + v_bytes, mte3_bytes=out_bytes)
    pools = {"l1": 524288, "ub": 196608, "l0a": 65536,
             "l0b": 65536, "l0c": 131072}
    single = apply_fusion(work_of(qk, soft, pv), pool_capacities=pools)
    split = apply_fusion(
        work_of(qk, soft, pv), pool_capacities=pools,
        core_count=2, per_core_rates={"cube": 1.0, "vector": 1.0},
        worker_count=2,
    )
    assert "attention_qk_softmax_pv" in split.applied_rules
    single_fused = fused_op(single)
    split_fused = fused_op(split)
    assert split_fused.flops == pytest.approx(single_fused.flops)
    assert split_fused.mte2_bytes == pytest.approx(single_fused.mte2_bytes)
    assert split_fused.hbm_bytes == pytest.approx(single_fused.hbm_bytes)
    meta = dict(split_fused.metadata)
    assert meta["worker_count"] == 2
    assert meta["worker_partition"] == "contiguous_heads"
    stages = meta["fusion_stages"]
    buffer_ids = [buffer["id"] for buffer in meta["buffer_plan"]]
    assert len(buffer_ids) == len(set(buffer_ids))
    for buffer in meta["buffer_plan"]:
        uses = {buffer["created_stage"], *buffer["consumers"]}
        workers = {stages[index].get("worker_id")
                   for index in uses
                   if stages[index].get("core_assignment", {}).get("count", 0)}
        assert len(workers) == 1
    assert len(set(meta["core_contract"]["worker_cores"].values())) == 2


def test_core_assignment_is_one_tile_one_core_not_die_peak():
    from mindie_sim.a3_fusion_common import (
        PipelinePlan, assign_stage_cores, core_group_makespan_ms,
        stage_core_service_ms,
    )

    # Every stage of one fused pipeline stays on one worker because producer
    # and consumer share local buffers.  Independent pipelines may use
    # different fixed workers.
    shared = PipelinePlan(kind="shared_pipeline")
    for index in range(16):
        shared.add_stage(f"tile{index}", "tile", deps=(), flops=100.0)
    shared_contract = assign_stage_cores(
        shared, core_count=16, per_core_rates={"cube": 1.0},
    )
    assert shared_contract["all_stages_same_worker"] is True
    assert {stage["core_assignment"]["core_ids"][0]
            for stage in shared.stages} == {0}
    assert all(
        stage["execution_width"] == {
            "aic_cores": 1, "aiv_cores": 0,
            "scope": "per_die", "partition": "single_core_tile",
        }
        for stage in shared.stages
    )

    workers = set()
    for index in range(16):
        plan = PipelinePlan(kind="independent_tile")
        plan.add_stage("tile", "tile", deps=(), flops=100.0)
        contract = assign_stage_cores(
            plan, core_count=16, per_core_rates={"cube": 1.0},
            core_base=index,
        )
        workers.add(contract["worker_cores"]["0"])
        assert stage_core_service_ms(
            plan.stages[0], {"cube": 1.0}
        ) == pytest.approx(100.0)
        assert core_group_makespan_ms(plan, {"cube": 1.0}) == pytest.approx(100.0)
    assert len(workers) == 16

    single = PipelinePlan(kind="single_core")
    single.add_stage("one_tile", "tile", deps=(), flops=1600.0)
    assign_stage_cores(single, core_count=16, per_core_rates={"cube": 1.0})
    assert stage_core_service_ms(
        single.stages[0], {"cube": 1.0}
    ) == pytest.approx(1600.0)
    assert single.stages[0]["execution_width"]["aic_cores"] == 1


def test_large_norm_build_is_bounded_and_finite_k_events_conserve_work():
    import time
    m, n, k = 512, 13824, 5120
    x_bytes, w_bytes, y_bytes = 2 * m * k, 2 * k * n, 2 * m * n
    norm = op("norm", "rmsnorm", (m, k), vector_ops=5 * m * k,
              hbm_bytes=2 * x_bytes, mte2_bytes=x_bytes, mte3_bytes=x_bytes)
    gemm = op("gemm", "qkv_projection", (m, n, k), ("norm",),
              flops=2 * m * n * k,
              hbm_bytes=x_bytes + w_bytes + y_bytes,
              mte2_bytes=x_bytes + w_bytes, mte3_bytes=y_bytes)
    pools = {"l1": 524288, "ub": 196608, "l0a": 65536,
             "l0b": 65536, "l0c": 131072}
    started = time.perf_counter()
    result = apply_fusion(work_of(norm, gemm), pool_capacities=pools)
    build_sec = time.perf_counter() - started
    assert "rmsnorm_gemm_epilogue" in result.applied_rules
    fused = fused_op(result)
    meta = dict(fused.metadata)
    assert build_sec < 25.0
    assert len(meta["fusion_stages"]) < 400000
    assert all(
        stage["extents"].get("compact_inner_loop") is not True
        for stage in meta["fusion_stages"]
    )
    weight_loads = [
        stage for stage in meta["fusion_stages"]
        if stage["kind"] == "weight_load"
    ]
    assert weight_loads
    for stage in weight_loads:
        created = sum(item["bytes"] for item in stage["buffers"]["create"])
        assert stage["mte2_bytes"] <= created + 1e-9
        extent = stage["extents"]["k"]
        assert extent[1] <= meta["tiling"]["k_tile"]
    assert sum(stage["flops"] for stage in meta["fusion_stages"]) == pytest.approx(
        gemm.flops
    )
    assert sum(stage["mte2_bytes"] for stage in meta["fusion_stages"]) == (
        pytest.approx(x_bytes + (m // meta["tiling"]["row_tile"]) * w_bytes)
    )

def test_placeholder_contractions_are_removed_without_launch_discount():
    conv = op("conv", "linear_conv", (8, 16), vector_ops=128.0,
              hbm_bytes=512, mte2_bytes=256, mte3_bytes=256)
    state = op("state", "linear_state", (8, 16), ("conv",), vector_ops=64.0,
               hbm_bytes=384, mte2_bytes=192, mte3_bytes=192)
    embed = op("embed", "vision_patch_embed", (4, 8), vector_ops=64.0,
               hbm_bytes=256, mte2_bytes=128, mte3_bytes=128)
    project = op("project", "vision_projector", (4, 8), ("embed",),
                 flops=128.0, hbm_bytes=256, mte2_bytes=128, mte3_bytes=128)
    gate_up0 = op("gu0", "moe_gate_up", (8, 16), flops=256.0,
                  hbm_bytes=512, mte2_bytes=256, mte3_bytes=256)
    down0 = op("d0", "moe_down", (8, 16), ("gu0",), flops=256.0,
               hbm_bytes=512, mte2_bytes=256, mte3_bytes=256)
    gate_up1 = op("gu1", "moe_gate_up", (8, 16), flops=256.0,
                  hbm_bytes=512, mte2_bytes=256, mte3_bytes=256)
    down1 = op("d1", "moe_down", (8, 16), ("gu1",), flops=256.0,
               hbm_bytes=512, mte2_bytes=256, mte3_bytes=256)
    cases = [
        ("linear", (conv, state)),
        ("vision", (embed, project)),
        ("moe", (gate_up0, down0, gate_up1, down1)),
    ]
    for label, original in cases:
        result = apply_fusion(work_of(*original), scratch_capacity_bytes=None)
        assert result.applied_rules == (), label
        assert result.work.ops == original, label
        assert all(item.coverage != "fused" for item in result.work.ops)
        assert result.changes["launch"] == 0
        assert result.changes["flops"] == 0
        assert result.changes["hbm_bytes"] == 0
        assert result.changes["mte2_bytes"] == 0
        assert result.changes["vector_ops"] == 0
        assert result.witness["fused_tasks"] == {}
        assert not any(item.get("eligible") for item in result.considered_rules)


def placement_kernel(name, *, stage, width=1, deps=()):
    base = op(name, "attention_fused", (1,), deps, coverage="fused")
    stages = tuple(
        {
            "stage_index": index,
            "name": f"{name}.stage{index}",
            "worker_id": index,
            "core_assignment": {
                "core_ids": (0,), "count": 1, "worker_id": index,
            },
        }
        for index in range(width)
    )
    meta = {
        "physical_feasible": True,
        "worker_count": width,
        "worker_cores": {str(index): index for index in range(width)},
        "core_contract": {
            "worker_cores": {str(index): index for index in range(width)},
            "workers_with_work": width,
        },
        "fusion_stages": stages,
    }
    return replace(base, stage=stage, metadata=tuple(sorted(meta.items())))


def test_full_work_placement_stage_round_robin_and_fixed_pin():
    ops = [
        placement_kernel("s0a", stage=0),
        placement_kernel("s0b", stage=0),
        placement_kernel("s0c", stage=0),
    ]
    out, placement = place_full_work_fusions(
        ops, core_count=2, core_base=None, fixed_core_base=False,
    )
    assert [dict(item.metadata)["full_work_placement"]["bundle_start"]
            for item in out] == [0, 1, 0]
    assert [item.op_id for item in out] == ["s0a", "s0b", "s0c"]
    assert "s0a" in next(item for item in out if item.op_id == "s0c").deps
    # More kernels than cores: the cursor wraps and reuses all bundles.
    many = [
        placement_kernel(f"m{index}", stage=0) for index in range(5)
    ]
    out, _ = place_full_work_fusions(
        many, core_count=2, core_base=None, fixed_core_base=False,
    )
    starts = [dict(item.metadata)["full_work_placement"]["bundle_start"]
              for item in out]
    assert starts == [0, 1, 0, 1, 0]
    # Stage1 gets an independent cursor and does not serialize on stage0 cores.
    stages = [
        placement_kernel("p0a", stage=0),
        placement_kernel("p0b", stage=0),
        placement_kernel("p1a", stage=1),
        placement_kernel("p1b", stage=1),
    ]
    out, _ = place_full_work_fusions(
        stages, core_count=2, core_base=None, fixed_core_base=False,
    )
    by_id = {item.op_id: item for item in out}
    assert "p0a" not in by_id["p1a"].deps
    assert "p0b" not in by_id["p1b"].deps
    assert [dict(by_id[key].metadata)["full_work_placement"]["bundle_start"]
            for key in ("p0a", "p0b", "p1a", "p1b")] == [0, 1, 0, 1]
    # Fixed pin keeps all stages on the requested core and serializes reuse.
    out, fixed = place_full_work_fusions(
        stages, core_count=2, core_base=0, fixed_core_base=True,
    )
    by_id = {item.op_id: item for item in out}
    assert "p0a" in by_id["p0b"].deps
    assert "p1a" in by_id["p1b"].deps
    assert "p0a" not in by_id["p1a"].deps
    for item in out:
        assert dict(item.metadata)["core_contract"]["worker_cores"] == {"0": 0}


def test_full_work_placement_mixed_width_wrap_waits_all_shared_cores():
    ops = [
        placement_kernel("b0", stage=0, width=1),
        placement_kernel("b1", stage=0, width=1),
        placement_kernel("b2", stage=0, width=3),
    ]
    out, placement = place_full_work_fusions(
        ops, core_count=3, core_base=None, fixed_core_base=False,
    )
    by_id = {item.op_id: item for item in out}
    assert dict(by_id["b2"].metadata)["core_contract"]["worker_cores"] == {
        "0": 0, "1": 1, "2": 2,
    }
    assert "b0" in by_id["b2"].deps
    assert "b1" in by_id["b2"].deps
    assert {edge["from"] for edge in placement["added_serialization_edges"]} == {
        "b0", "b1",
    }
    fixed_out, _ = place_full_work_fusions(
        ops, core_count=3, core_base=0, fixed_core_base=True,
    )
    fixed = {item.op_id: item for item in fixed_out}
    assert "b1" in fixed["b2"].deps
    assert "b0" not in fixed["b2"].deps  # b1 already completes after b0
    assert "b0" in fixed["b1"].deps
