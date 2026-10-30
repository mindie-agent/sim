"""Independent structure, traffic and ownership controls for staged GQA decode.

The expected ledgers are re-derived from request geometry, causal horizons and
the declared split/tile policy.  Synthetic service rates make trends and
dependencies observable; they do not establish measured hardware accuracy.
"""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from mindie_sim.prediction.attention_decode import (
    RECIPE_ID,
    StagedGQADecode,
    causal_pairs,
)
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


def hardware(*, units=4):
    return {
        "machine": {"name": "synthetic-gqa-decode", "num_devices": 1,
                    "device": {
                        "memory_bytes": 1 << 30,
                        "compute_units": units,
                        "compute_peaks": {"bf16": 8e12, "fp32": 4e12},
                        "rates": {"memory": 1e9, "vector": 1e9,
                                  "mte1": 1e9, "mte2": 1e9, "mte3": 1e9},
                        "buffers": {"l1": 1 << 20, "l0a": 1 << 18,
                                    "l0b": 1 << 18, "l0c": 1 << 19,
                                    "ub": 1 << 20},
                        "buffer_units": {"l1": units, "l0a": units,
                                         "l0b": units, "l0c": units,
                                         "ub": 2 * units},
                        "resource_units": {"vector": 2 * units, "mte1": units,
                                           "mte2": units, "mte3": units},
                    }},
        "model": {"name": "geometry", "layers": 1, "hidden_size": 8,
                  "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
                  "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1,
                                   "output_tokens": 1}]},
        "runtime": {},
    }


def make_op(requests, *, heads=4, kv_heads=2, dim=8,
            recipe=RECIPE_ID, **metadata_overrides):
    metadata = {
        "requests": tuple(requests), "query_heads": heads,
        "kv_heads": kv_heads, "head_dim": dim,
        "query_dtype": "bf16", "kv_dtype": "bf16",
        "causal": True, "window": 0, "cp": 1,
    }
    metadata.update(metadata_overrides)
    tokens = sum(query for query, _ in requests)
    length = max((context + query for query, context in requests), default=0)
    shape = (tokens, length, heads, dim)
    return OpSpec(
        "gqa_decode", "attention", "decode", 0, 0,
        tuple(f"r{index}" for index in range(len(requests))), shape, shape,
        shape, shape, recipe=recipe, dtype="bf16",
        metadata=(("gqa_decode", metadata),),
    )


def split_ranges(tiles, requested):
    count = min(requested, tiles)
    base, extra = divmod(tiles, count)
    ranges, start = [], 0
    for index in range(count):
        size = base + (1 if index < extra else 0)
        ranges.append((start, start + size))
        start += size
    return tuple(ranges)


def expected_traffic(requests, *, heads, kv_heads, dim, key_tile, query_tile,
                     kv_splits, query_bytes=2, kv_bytes=2):
    """Independent HBM/traffic ledger from geometry and causal horizons."""
    group = heads // kv_heads
    q_read = kv_read = partial_write = partial_read = out_write = 0
    flops = vector = 0
    outputs = 0
    for query, context in requests:
        if not query:
            continue
        rows = query * group
        tiles = math.ceil((context + query) / key_tile)
        ranges = split_ranges(tiles, kv_splits)
        for _kv_head in range(kv_heads):
            for row_start in range(0, rows, query_tile):
                row_end = min(rows, row_start + query_tile)
                token_first, token_last = row_start // group, (row_end - 1) // group
                horizon = context + token_last + 1
                cutoff = math.ceil(horizon / key_tile)
                contributors = [span for span in ranges
                                if min(span[1], cutoff) > span[0]]
                assert contributors
                outputs += 1
                pairs = sum(
                    (min(row_end, (token + 1) * group)
                     - max(row_start, token * group)) * (context + token + 1)
                    for token in range(token_first, token_last + 1)
                )
                row_tiles = sum(
                    (min(row_end, (token + 1) * group)
                     - max(row_start, token * group))
                    * math.ceil((context + token + 1) / key_tile)
                    for token in range(token_first, token_last + 1)
                )
                q_read += len(contributors) * (row_end - row_start) * dim * query_bytes
                kv_read += horizon * 2 * dim * kv_bytes
                out_write += (row_end - row_start) * dim * query_bytes
                flops += 4 * dim * pairs
                vector += 7 * pairs + (3 * dim + 10) * row_tiles
                if len(contributors) > 1:
                    partial = (len(contributors) * (row_end - row_start)
                               * (dim * 4 + 2 * 4))
                    partial_write += partial
                    partial_read += partial
                    vector += (len(contributors) * (row_end - row_start)
                               * (3 * dim + 9)
                               + (row_end - row_start) * (dim + 1))
                else:
                    vector += (row_end - row_start) * (dim + 1)
    return {
        "outputs": outputs,
        "q_read": q_read, "kv_read": kv_read,
        "partial_write": partial_write, "partial_read": partial_read,
        "output_write": out_write,
        "hbm": q_read + kv_read + partial_write + partial_read + out_write,
        "mte2": q_read + kv_read + partial_read,
        "mte3": partial_write + out_write,
        "flops": flops, "vector": vector,
    }


def run(op, recipe=None, *, units=4, ready=0.0, external=None, raw_case=None):
    recipe = recipe or StagedGQADecode()
    case = parse_case(raw_case or hardware(units=units))
    execution = Execution(case, CostModel(case),
                          recipes=RecipeBook({RECIPE_ID: recipe}),
                          trace_limit=20000)
    prepared, _ = execution.prepare(op)
    context = execution._context(prepared, 0, "none")
    start = execution.engine.submit("input_ready", release=ready)
    if external:
        execution.engine.submit(
            "external", external["work"], placement=external["placement"],
            owner="external",
        )
    end = recipe.lower(context).start(execution.engine, "gqa", start)
    before = {}
    if ready:
        execution.engine.advance(horizon=ready)
        before = {name: value for name, value in execution.engine.serviced.items()
                  if value}
    execution.engine.advance(until=(end,))
    return execution.engine, end, before, context, recipe


def test_qk_and_pv_include_the_causal_triangular_tail():
    query, context = 4, 5
    op = make_op(((query, context),), heads=4, kv_heads=2, dim=8)
    recipe = StagedGQADecode(key_tile=4)
    work = recipe.work(op, include_mte1=True)
    # 6+7+8+9 valid keys in one query head; QK and PV are both 2*D*pairs.
    assert causal_pairs(query, context) == 30
    assert work.matrix == {"bf16": 4 * 30 * 4 * 8}

    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(
        expected_traffic(((query, context),), heads=4, kv_heads=2, dim=8,
                         key_tile=4, query_tile=64, kv_splits=1)["hbm"])
    assert engine.serviced["mte1@r0:s0"] == work.resources["mte1"]


@pytest.mark.parametrize("splits", (1, 2))
def test_hbm_for_ragged_batch_and_split_partial_roundtrip(splits):
    requests = ((1, 8), (2, 3))
    op = make_op(requests, heads=4, kv_heads=2, dim=8)
    recipe = StagedGQADecode(key_tile=4, kv_splits=splits)
    expected = expected_traffic(
        requests, heads=4, kv_heads=2, dim=8, key_tile=4,
        query_tile=64, kv_splits=splits,
    )
    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    assert engine.serviced["memory@r0:s0"] == pytest.approx(expected["hbm"])
    assert engine.serviced["mte2@r0:s0"] == pytest.approx(expected["mte2"])
    assert engine.serviced["mte3@r0:s0"] == pytest.approx(expected["mte3"])
    if splits == 1:
        assert expected["partial_write"] == expected["partial_read"] == 0
    else:
        assert expected["partial_write"] > 0
        assert expected["partial_read"] == expected["partial_write"]
    # Final output is written once, independent of split count.
    assert expected["output_write"] == 4 * 8 * 2 + 2 * 4 * 8 * 2


def test_zero_context_and_gqa_head_reuse_do_not_duplicate_kv():
    zero = ((1, 0), (3, 0))
    op = make_op(zero, heads=4, kv_heads=2, dim=8)
    recipe = StagedGQADecode(key_tile=2, kv_splits=2)
    expected = expected_traffic(
        zero, heads=4, kv_heads=2, dim=8, key_tile=2,
        query_tile=64, kv_splits=2,
    )
    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    # Only the three-token request spans two key tiles; the one-token request
    # keeps a direct final store.  No padded context capacity is charged.
    assert expected["partial_write"] > 0
    assert engine.serviced["memory@r0:s0"] == pytest.approx(expected["hbm"])

    requests = ((1, 8),)
    narrow = expected_traffic(
        requests, heads=4, kv_heads=2, dim=8, key_tile=4,
        query_tile=64, kv_splits=2,
    )
    wide = expected_traffic(
        requests, heads=8, kv_heads=2, dim=8, key_tile=4,
        query_tile=64, kv_splits=2,
    )
    assert wide["kv_read"] == narrow["kv_read"]
    assert wide["q_read"] == 2 * narrow["q_read"]
    assert wide["output_write"] == 2 * narrow["output_write"]


def test_no_service_before_input_readiness():
    op = make_op(((1, 9),), heads=2, kv_heads=1, dim=8)
    engine, end, before, _, _ = run(
        op, StagedGQADecode(key_tile=4, kv_splits=2), ready=1.25,
    )
    assert before == {}
    assert end.time >= 1.25


def test_live_tasks_and_allocations_are_released_and_peak_is_bounded():
    short = make_op(((1, 32),), heads=2, kv_heads=1, dim=8)
    long = make_op(((1, 4096),), heads=2, kv_heads=1, dim=8)
    recipe = StagedGQADecode(key_tile=16)
    short_engine, short_end, _, _, _ = run(short, recipe)
    long_engine, long_end, _, _, _ = run(long, recipe)
    for engine in (short_engine, long_engine):
        assert not engine._tasks
        assert not engine._promises
        assert not engine.allocations
        assert all(value == 0 for value in engine.allocated.values())
    assert long_end.time > short_end.time
    assert long_engine.peak_tasks <= short_engine.peak_tasks + 2
    for pool, peak in short_engine.peak_allocated.items():
        assert long_engine.peak_allocated[pool] <= peak + 1


def test_owner_pinning_cannot_borrow_idle_per_core_units():
    op = make_op(((1, 16),), heads=1, kv_heads=1, dim=8)
    recipe = StagedGQADecode(key_tile=32)
    one_engine, one_end, _, _, _ = run(op, recipe, units=1)
    four_engine, four_end, _, _, _ = run(op, recipe, units=4)
    assert four_end.time > 2.5 * one_end.time

    vector = "vector@r0:s0"
    external = {"work": {vector: 1e7},
                "placement": {vector: (2, 3, 4, 5, 6, 7)}}
    engine, end, _, _, _ = run(op, recipe, units=4, external=external)
    # The external work is still running on the other owners when this task
    # completes; it cannot change the pinned task's completion time.
    assert end.time == pytest.approx(four_end.time, rel=1e-12, abs=1e-12)
    assert engine.serviced[vector] > 0


def test_split_merge_waits_for_every_partial_producer():
    op = make_op(((1, 17),), heads=2, kv_heads=1, dim=8)
    recipe = StagedGQADecode(key_tile=4, kv_splits=2)
    engine, end, _, _, _ = run(op, recipe, units=2)
    rows = list(engine.trace)
    partials = [row for row in rows if row["task"].endswith(":partial_store")]
    merges = [row for row in rows if row["task"].endswith(":merge_load")]
    assert len(partials) == 2
    assert len(merges) == 2
    assert min(row["start_ms"] for row in merges) >= max(
        row["end_ms"] for row in partials)
    assert not any(row["task"].endswith(":final_store") for row in rows)
    assert end.done

    single = StagedGQADecode(key_tile=4, kv_splits=1)
    engine1, _, _, _, _ = run(op, single, units=2)
    assert not any(row["task"].endswith(":partial_store")
                   for row in engine1.trace)
    assert not any(":merge" in row["task"] for row in engine1.trace)


def test_multiple_query_tiles_keep_one_output_write_per_tile():
    requests = ((4, 6),)
    op = make_op(requests, heads=4, kv_heads=1, dim=8)
    recipe = StagedGQADecode(key_tile=4, query_tile=4, kv_splits=2)
    expected = expected_traffic(
        requests, heads=4, kv_heads=1, dim=8, key_tile=4,
        query_tile=4, kv_splits=2,
    )
    engine, end, _, _, _ = run(op, recipe)
    assert end.done
    assert expected["outputs"] == 4
    assert engine.serviced["memory@r0:s0"] == pytest.approx(expected["hbm"])
    stores = [row for row in engine.trace
              if row["task"].endswith(":final_store")
              or row["task"].endswith(":merge_store")]
    assert len(stores) == expected["outputs"]
    assert sum(row["demand"].get("memory@r0:s0", 0) for row in stores) == (
        pytest.approx(expected["output_write"])
    )


@pytest.mark.parametrize("overrides", (
    {"window": 1}, {"cp": 2}, {"query_heads": 3},
))
def test_unsupported_domain_is_rejected(overrides):
    op = make_op(((1, 8),), heads=4, kv_heads=2, dim=8, **overrides)
    with pytest.raises(ModelingError):
        StagedGQADecode().work(op)


def test_split_read_pipes_keep_cube_operands_and_vector_merge_separate():
    raw = hardware(units=2)
    device = raw["machine"]["device"]
    device["rates"].pop("mte2")
    device["rates"].update(mte2_aic=1e9, mte2_aiv=1e9)
    device["resource_units"].pop("mte2")
    device["resource_units"].update(mte2_aic=2, mte2_aiv=4)
    op = make_op(((2, 9),), heads=4, kv_heads=2, dim=8)
    recipe = StagedGQADecode(key_tile=4, kv_splits=2)
    expected = expected_traffic(
        ((2, 9),), heads=4, kv_heads=2, dim=8, key_tile=4,
        query_tile=64, kv_splits=2)
    engine, end, _, _, _ = run(op, recipe, units=2, raw_case=raw)
    assert end.done
    assert engine.serviced["mte2_aic@r0:s0"] == (
        expected["q_read"] + expected["kv_read"])
    assert engine.serviced["mte2_aiv@r0:s0"] == expected["partial_read"]
    probability_bytes = (10 + 11) * 4 * 2
    assert engine.serviced["mte1@r0:s0"] == pytest.approx(
        expected["q_read"] + expected["kv_read"] + probability_bytes)
    for row in engine.trace:
        for resource in ("mte2_aic@r0:s0", "mte2_aiv@r0:s0"):
            if row["demand"].get(resource):
                assert resource in row["placement"]
