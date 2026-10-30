"""Independent math, address, ownership and launch oracles for MLA decode.

The tests re-derive split ranges, per-CTA traffic/FLOPs, the online-LSE merge
and physical-unit placement from the cited source lines rather than importing
the planner's private formulas.  A finite, conserved program is structural
verification; the NumPy attention, hand-computed ledgers, per-CTA trace and
fixed-per-SM counterexamples below check the mechanism itself.  No test claims
capture or device-latency accuracy.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.operators import mla_decode
from mindie_sim.prediction.program import (
    KernelProgram,
    ParallelProgram,
    PipelineProgram,
    RecipeBook,
    RecipeContext,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView

CAMPAIGN = (
    Path(__file__).resolve().parents[2]
    / "tests/fixtures/operator_cases/mla_decode.json"
)
SELECTED = "triton_decode_grouped_mla"

# Selected non-HIP source constants (triton_decode_attention.py lines 491-519).
BLOCK_H = 16
BLOCK_N = 32
INDEX_BYTES = 4
PROGRAM_BYTES = 2
ACCUM_BYTES = 4

NAMES = {
    "cube": "matrix@r0:s0",
    "vector": "vector@r0:s0",
    "hbm": "memory@r0:s0",
    "mte2": "mte2@r0:s0",
    "mte3": "mte3@r0:s0",
}


def campaign_cases():
    return json.loads(CAMPAIGN.read_text())["cases"]


def base_params(**overrides):
    params = {
        "contexts": [17],
        "heads": 16,
        "key_heads": 1,
        "latent_dim": 512,
        "rope_dim": 64,
        "page_size": 16,
        "num_kv_splits": 2,
        "dtype": "bf16",
        "cache_dtype": "bf16",
        "implementation": SELECTED,
        "is_mla": True,
        "is_hip": False,
        "logit_cap": 0.0,
    }
    params.update(overrides)
    return params


def hardware_case(*, units=8, copy_engines=False, launch_us=0, bf16_peak=1e14,
                  buffers=None, buffer_units=None, rate_scale=1.0):
    rates = {"memory": 1e12 * rate_scale, "vector": 1e12 * rate_scale}
    if copy_engines:
        rates.update(read=1e12 * rate_scale, write=1e12 * rate_scale)
    device = {
        "memory_bytes": 64 * 1024**3,
        "compute_units": units,
        "compute_peaks": {"bf16": bf16_peak, "fp32": 2.5e13},
        "rates": rates,
        "launch_us": launch_us,
    }
    if buffers:
        device["buffers"] = dict(buffers)
    if buffer_units:
        device["buffer_units"] = dict(buffer_units)
    return {
        "machine": {"name": "synthetic-mla-oracle", "num_devices": 1,
                    "device": device},
        "model": {
            "name": "geometry", "layers": 1, "hidden_size": 8,
            "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
            "intermediate_size": 16, "vocab_size": 16,
        },
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1,
                                   "output_tokens": 1}]},
        "runtime": {},
    }


def geometry(units=8):
    return Geometry(units, {}, {})


def child_plans(op, units=8):
    stage1, stage2 = mla_decode.kernels(op)
    return (
        stage1,
        stage2,
        mla_decode.build(stage1, geometry(units)),
        mla_decode.build(stage2, geometry(units)),
    )


def work_ledger(plan):
    resources = dict(plan.work.resources)
    return {
        "flops": dict(plan.work.matrix).get("bf16", 0),
        "hbm": resources.get("hbm", 0),
        "mte2": resources.get("mte2", 0),
        "mte3": resources.get("mte3", 0),
        "vector": resources.get("vector", 0),
    }


# ---------------------------------------------------------------------------
# Independent source oracles: cdiv split spans, block dispatch and ledgers.
# ---------------------------------------------------------------------------


def split_partition(length, splits, block_n=BLOCK_N):
    """Independent oracle for cdiv split ranges and restarted tile loops."""
    chunk = -(-length // splits)
    spans = []
    for index in range(splits):
        start = index * chunk
        end = min(start + chunk, length)
        if end > start:
            spans.append((start, end))
    tiles = sum(-(-(end - start) // block_n) for start, end in spans)
    return chunk, spans, tiles


def oracle_blocks(latent, rope):
    # triton_decode_attention.py lines 491-504, independently restated.
    return {
        (512, 64): (512, 64, 512),
        (256, 32): (256, 32, 256),
    }[(latent, rope)]


def oracle_work(params):
    """Closed-form per-kernel ledger derived from source lines 279-467/575-639."""
    seq_lens = tuple(params["contexts"])
    heads = params["heads"]
    splits = params["num_kv_splits"]
    latent, rope = params["latent_dim"], params["rope_dim"]
    block_dmodel, block_dpe, block_dv = oracle_blocks(latent, rope)
    qk_dim = latent + rope
    valid_block_h = min(BLOCK_H, heads)
    head_blocks = -(-heads // valid_block_h)

    tile_vector = 7 * BLOCK_H * BLOCK_N + BLOCK_H * block_dv + 5 * BLOCK_H
    epilogue_vector = BLOCK_H * block_dv + 2 * BLOCK_H
    tile_flops = 2 * BLOCK_H * BLOCK_N * (block_dmodel + block_dpe + block_dv)
    merge_vector = 3 * block_dv + 7
    final_vector = 2 * block_dv + 3

    s1_read = s1_write = s1_vector = s1_flops = 0
    s2_read = s2_write = s2_vector = 0
    chunks, nonempty_counts, tile_counts = [], [], []
    for length in seq_lens:
        chunk, spans, tiles = split_partition(length, splits)
        nonempty = len(spans)
        chunks.append(chunk)
        nonempty_counts.append(nonempty)
        tile_counts.append(tiles)

        # Q is loaded by every launched split program, including empty ones.
        s1_read += splits * heads * qk_dim * PROGRAM_BYTES
        s1_read += head_blocks * splits * INDEX_BYTES       # B_Seqlen
        s1_read += head_blocks * nonempty * 2 * ACCUM_BYTES  # k_scale/v_scale
        # Page-id lane + one 576-column latent/PE K row per key, per head block.
        s1_read += head_blocks * length * INDEX_BYTES
        s1_read += head_blocks * length * qk_dim * PROGRAM_BYTES
        s1_write += nonempty * heads * (block_dv * ACCUM_BYTES + ACCUM_BYTES)
        s1_vector += head_blocks * (
            tiles * tile_vector + nonempty * epilogue_vector
        )
        s1_flops += head_blocks * tiles * tile_flops

        # Stage 2 reads only non-empty slots; every head writes o and lse.
        s2_read += heads * INDEX_BYTES
        s2_read += heads * nonempty * (block_dv * ACCUM_BYTES + ACCUM_BYTES)
        s2_write += heads * (block_dv * PROGRAM_BYTES + PROGRAM_BYTES)
        s2_vector += heads * (
            nonempty * merge_vector + final_vector
        )

    return {
        "stage1": {
            "flops": s1_flops,
            "hbm": s1_read + s1_write,
            "mte2": s1_read,
            "mte3": s1_write,
            "vector": s1_vector,
        },
        "stage2": {
            "flops": 0,
            "hbm": s2_read + s2_write,
            "mte2": s2_read,
            "mte3": s2_write,
            "vector": s2_vector,
        },
        "total": {
            "flops": s1_flops,
            "hbm": s1_read + s1_write + s2_read + s2_write,
            "mte2": s1_read + s2_read,
            "mte3": s1_write + s2_write,
            "vector": s1_vector + s2_vector,
        },
        "blocks": (block_dmodel, block_dpe, block_dv),
        "head_blocks": head_blocks,
        "valid_block_h": valid_block_h,
        "chunks": tuple(chunks),
        "nonempty": tuple(nonempty_counts),
        "tiles": tuple(tile_counts),
    }


def oracle_stage1_cta(params, request, head_block, split_id):
    """Independent per-CTA stage-1 ledger for one source program."""
    heads = params["heads"]
    splits = params["num_kv_splits"]
    length = params["contexts"][request]
    block_dmodel, block_dpe, block_dv = oracle_blocks(
        params["latent_dim"], params["rope_dim"]
    )
    qk_dim = params["latent_dim"] + params["rope_dim"]
    valid_block_h = min(BLOCK_H, heads)
    valid_h = min(BLOCK_H, heads - head_block * valid_block_h)
    chunk, _, _ = split_partition(length, splits)
    start = split_id * chunk
    end = min(start + chunk, length)
    span = max(0, end - start)
    tiles = -(-span // BLOCK_N) if span else 0
    tile_loads = []
    for index in range(tiles):
        valid_n = min(BLOCK_N, end - (start + index * BLOCK_N))
        tile_loads.append(
            valid_n * (INDEX_BYTES + qk_dim * PROGRAM_BYTES)
            + (2 * ACCUM_BYTES if index == 0 else 0)
        )
    return {
        "valid_h": valid_h,
        "tiles": tiles,
        "q_bytes": valid_h * qk_dim * PROGRAM_BYTES + INDEX_BYTES,
        "tile_loads": tuple(tile_loads),
        "flops": tiles * 2 * BLOCK_H * BLOCK_N
        * (block_dmodel + block_dpe + block_dv),
        "vector": tiles
        * (7 * BLOCK_H * BLOCK_N + BLOCK_H * block_dv + 5 * BLOCK_H)
        + (BLOCK_H * block_dv + 2 * BLOCK_H if span else 0),
        "store_bytes": valid_h * (block_dv * ACCUM_BYTES + ACCUM_BYTES),
    }


# ---------------------------------------------------------------------------
# NumPy dense-attention and split-LSE merge oracle.
# ---------------------------------------------------------------------------


def _qk(q_nope, q_pe, k_latent, k_pe, scale):
    return (q_nope @ k_latent.T + q_pe @ k_pe.T) * scale


def dense_latent_attention(q_nope, q_pe, k_latent, k_pe, scale):
    scores = _qk(q_nope, q_pe, k_latent, k_pe, scale)
    maximum = scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores - maximum)
    normalizer = weights.sum(axis=-1, keepdims=True)
    output = (weights @ k_latent) / normalizer
    lse = (maximum + np.log(normalizer))[:, 0]
    return output, lse


def split_latent_attention(q_nope, q_pe, k_latent, k_pe, scale, splits):
    """Stage-1 partials then stage-2 online merge, including empty splits."""
    length = k_latent.shape[0]
    chunk = -(-length // splits)
    partials = []
    for split in range(splits):
        start, end = split * chunk, min((split + 1) * chunk, length)
        if end <= start:
            continue
        scores = _qk(q_nope, q_pe, k_latent[start:end], k_pe[start:end], scale)
        maximum = scores.max(axis=-1)
        weights = np.exp(scores - maximum[:, None])
        normalizer = weights.sum(axis=-1)
        output = (weights @ k_latent[start:end]) / normalizer[:, None]
        partials.append((output, maximum + np.log(normalizer), start, end))

    assert sum(end - start for _, _, start, end in partials) == length
    acc = np.zeros_like(q_nope)
    e_sum = 0.0
    e_max = -np.inf
    for output, lse, _, _ in partials:
        new_max = np.maximum(lse, e_max)
        old_scale = np.exp(e_max - new_max)
        new_scale = np.exp(lse - new_max)
        acc = acc * old_scale[:, None] + new_scale[:, None] * output
        e_sum = e_sum * old_scale + new_scale
        e_max = new_max
    return acc / e_sum[:, None], e_max + np.log(e_sum)


@pytest.mark.parametrize("splits", [1, 2, 3, 5, 8, 17])
def test_numpy_split_lse_merge_matches_dense_latent_attention(splits):
    rng = np.random.default_rng(7301 + splits)
    length, heads, latent, rope = 17, 5, 12, 4
    q_nope = rng.normal(size=(heads, latent))
    q_pe = rng.normal(size=(heads, rope))
    k_latent = rng.normal(size=(length, latent))
    k_pe = rng.normal(size=(length, rope))
    dense, dense_lse = dense_latent_attention(q_nope, q_pe, k_latent, k_pe, 0.37)
    split, split_lse = split_latent_attention(
        q_nope, q_pe, k_latent, k_pe, 0.37, splits
    )
    np.testing.assert_allclose(split, dense, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(split_lse, dense_lse, rtol=1e-12, atol=1e-12)


def test_numpy_merge_covers_selected_512_64_geometry_and_empty_splits():
    rng = np.random.default_rng(9127)
    length, heads, latent, rope, splits = 3, 4, 512, 64, 7
    q_nope = rng.normal(size=(heads, latent))
    q_pe = rng.normal(size=(heads, rope))
    k_latent = rng.normal(size=(length, latent))
    k_pe = rng.normal(size=(length, rope))
    dense, dense_lse = dense_latent_attention(
        q_nope, q_pe, k_latent, k_pe, 0.125
    )
    split, split_lse = split_latent_attention(
        q_nope, q_pe, k_latent, k_pe, 0.125, splits
    )
    np.testing.assert_allclose(split, dense, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(split_lse, dense_lse, rtol=1e-12, atol=1e-12)
    # Four empty split slots exercise the stage-2 branch that must skip them.
    assert len(split_partition(length, splits)[1]) == 3


def test_latent_pv_alias_is_required_and_matches_no_separate_v_buffer():
    rng = np.random.default_rng(333)
    length, heads, latent, rope = 9, 3, 16, 5
    q_nope = rng.normal(size=(heads, latent))
    q_pe = rng.normal(size=(heads, rope))
    k_latent = rng.normal(size=(length, latent))
    k_pe = rng.normal(size=(length, rope))
    other_v = rng.normal(size=(length, latent))
    scores = _qk(q_nope, q_pe, k_latent, k_pe, 0.5)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    latent_pv = weights @ k_latent       # line 432: v = tl.trans(k)
    dense, _ = dense_latent_attention(q_nope, q_pe, k_latent, k_pe, 0.5)
    np.testing.assert_allclose(latent_pv, dense, rtol=1e-12, atol=1e-12)
    assert not np.allclose(latent_pv, weights @ other_v)


# ---------------------------------------------------------------------------
# Per-kernel hand-computed ledgers and empty-split behavior.
# ---------------------------------------------------------------------------


def test_hand_computed_per_kernel_ledgers_for_single_token_split_one():
    params = base_params(contexts=[1], num_kv_splits=1)
    parent = mla_decode.make_op(params)
    stage1, stage2, plan1, plan2 = child_plans(parent)
    assert work_ledger(plan1) == {
        "flops": 1114112, "hbm": 52432, "mte2": 19600,
        "mte3": 32832, "vector": 20080,
    }
    assert work_ledger(plan2) == {
        "flops": 0, "hbm": 49312, "mte2": 32896,
        "mte3": 16416, "vector": 41120,
    }
    # The two kernels sum to the independently derived composite ledger.
    total = {key: work_ledger(plan1)[key] + work_ledger(plan2)[key]
             for key in work_ledger(plan1)}
    assert total == {
        "flops": 1114112, "hbm": 101744, "mte2": 52496,
        "mte3": 49248, "vector": 61200,
    }
    assert dict(parent.metadata)["plan"]["stage1_grid"] == (1, 1, 1)


def test_hand_computed_tail_and_empty_split_ledgers():
    # 17 keys split 2 ways: 9 + 8 keys => two BLOCK_N tiles, not one.
    params = base_params(contexts=[17], num_kv_splits=2)
    parent = mla_decode.make_op(params)
    _, _, plan1, plan2 = child_plans(parent)
    oracle = oracle_work(params)
    assert work_ledger(plan1) == oracle["stage1"]
    assert work_ledger(plan2) == oracle["stage2"]
    assert work_ledger(plan1)["flops"] == 2 * 1114112

    # 1 key split 8 ways: one non-empty split does the MMA and scratch store;
    # seven empty split programs still load Q and B_Seqlen.
    empty = base_params(contexts=[1], num_kv_splits=8)
    one = base_params(contexts=[1], num_kv_splits=1)
    empty_plan = mla_decode.build(
        mla_decode.kernels(mla_decode.make_op(empty))[0], geometry()
    )
    one_plan = mla_decode.build(
        mla_decode.kernels(mla_decode.make_op(one))[0], geometry()
    )
    empty_ledger, one_ledger = work_ledger(empty_plan), work_ledger(one_plan)
    assert empty_ledger["flops"] == one_ledger["flops"]
    assert empty_ledger["vector"] == one_ledger["vector"]
    assert empty_ledger["mte3"] == one_ledger["mte3"]
    # Seven extra launched splits: Q row block + one B_Seqlen each; no
    # k_scale/v_scale and no K/PV/scratch for empty slots.
    assert empty_ledger["mte2"] - one_ledger["mte2"] == (
        7 * (16 * 576 * 2 + 4)
    )
    metadata = dict(mla_decode.make_op(empty).metadata)
    assert metadata["plan"]["nonempty_splits"] == (1,)
    assert metadata["plan"]["block_tiles"] == (1,)


def test_split_partition_matches_cdiv_and_tile_oracle_across_tails():
    for length in (1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 64, 65, 129):
        for splits in (1, 2, 3, 4, 5, 8, 16):
            chunk, spans, tiles = split_partition(length, splits)
            params = base_params(contexts=[length], num_kv_splits=splits)
            op = mla_decode.make_op(params)
            plan_meta = dict(op.metadata)["plan"]
            assert plan_meta["split_chunks"] == (chunk,)
            assert plan_meta["nonempty_splits"] == (len(spans),)
            assert plan_meta["block_tiles"] == (tiles,)
            _, _, plan1, _ = child_plans(op)
            assert work_ledger(plan1)["flops"] == (
                tiles * 2 * BLOCK_H * BLOCK_N * (512 + 64 + 512)
            )


# ---------------------------------------------------------------------------
# Campaign coverage against the independent per-kernel oracle.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("length,splits,heads", [
    (1, 8, 2),
    (17, 2, 16),
    (33, 4, 24),
    (65, 3, 8),
    (129, 8, 32),
])
def test_per_kernel_ledgers_match_oracle_across_tails_and_empty_splits(
    length, splits, heads
):
    params = base_params(contexts=[length], heads=heads, num_kv_splits=splits)
    op = mla_decode.make_op(params)
    _, _, plan1, plan2 = child_plans(op)
    oracle = oracle_work(params)
    assert work_ledger(plan1) == oracle["stage1"]
    assert work_ledger(plan2) == oracle["stage2"]


@pytest.mark.parametrize("case", campaign_cases(), ids=lambda case: case["id"])
def test_campaign_dispatch_and_work_match_independent_oracle(case):
    op = mla_decode.make_op(case["params"])
    dispatch = dict(op.metadata)["dispatch"]
    for key, expected in case["expected_dispatch"].items():
        assert dispatch[key] == expected
    assert dispatch["stage1_kernel"] == "_fwd_grouped_kernel_stage1"
    assert dispatch["stage2_kernel"] == "_fwd_kernel_stage2"
    assert dispatch["path"] == "grouped"
    assert dispatch["split_selection"] == "explicit"
    assert dispatch["pv_source"] == "latent_k_alias"
    assert dispatch["v_buffer_reads"] == 0
    assert dispatch["block_h"] == 16 and dispatch["block_n"] == 32
    assert dispatch["source_kernel_launches"] == 2
    assert dispatch["kernel_launches"] == (
        "stage1_grouped_decode", "stage2_split_merge"
    )
    assert dispatch["tile_order"] == "qk_softmax_pv"
    assert dispatch["empty_splits"] == "stage1_launches_stage2_skips"
    assert dispatch["split_reduction_owner"] == "batch_head"
    assert dispatch["num_kv_splits"] == case["params"]["num_kv_splits"]
    assert dict(op.metadata)["kernels"] == (
        "stage1_grouped_decode", "stage2_split_merge"
    )

    expected = oracle_work(case["params"])
    stage1, stage2, plan1, plan2 = child_plans(op)
    assert work_ledger(plan1) == expected["stage1"]
    assert work_ledger(plan2) == expected["stage2"]

    metadata = dict(op.metadata)
    plan_meta = metadata["plan"]
    assert plan_meta["head_blocks"] == expected["head_blocks"]
    assert plan_meta["block_tiles"] == expected["tiles"]
    assert plan_meta["nonempty_splits"] == expected["nonempty"]
    assert plan_meta["split_chunks"] == expected["chunks"]
    assert plan_meta["pages_per_request"] == tuple(
        math.ceil(length / case["params"]["page_size"])
        for length in case["params"]["contexts"]
    )
    assert plan_meta["page_index_entries"] == (
        expected["head_blocks"] * sum(case["params"]["contexts"])
    )
    assert plan_meta["stage1_grid"] == (
        len(case["params"]["contexts"]),
        expected["head_blocks"],
        case["params"]["num_kv_splits"],
    )
    assert plan_meta["stage2_grid"] == (
        len(case["params"]["contexts"]),
        case["params"]["heads"],
    )

    # Independent NumPy check of the split/merge path on a small shape drawn
    # from the same selected blocks is covered separately; here run the public
    # isolated path and require the executed service to equal each kernel plan.
    for copy_engines in (False, True):
        report = verify_operator(
            hardware_case(copy_engines=copy_engines), op, mla_decode.build,
            kernels=mla_decode.kernels,
        )
        assert report["status"] == "passed", (case["id"], report)
        assert report["work_mismatches"] == {}
        assert report["buffers_released"] and report["finite_completion"]
        rows = report["result"]["kernels"]
        assert len(rows) == 2
        assert {row["operator"]["op_id"] for row in rows} == {
            "mla_decode/stage1_grouped_decode", "mla_decode/stage2_split_merge"
        }
        runtime = sum(row["duration_ms"] for row in rows)
        assert runtime == pytest.approx(report["result"]["duration_ms"])
        for row, stage in zip(
            sorted(rows, key=lambda row: row["operator"]["op_id"]),
            (expected["stage1"], expected["stage2"]),
        ):
            assert row["operator"]["flops"] == stage["flops"]
            assert row["operator"]["hbm_bytes"] == stage["hbm"]
            assert row["operator"]["mte2_bytes"] == stage["mte2"]
            assert row["operator"]["mte3_bytes"] == stage["mte3"]
            assert row["operator"]["vector_ops"] == stage["vector"]
            for name, delta in row["service_delta"].items():
                base = row["declared_service"].get(name, 0.0)
                assert abs(delta) <= 1e-9 * max(1.0, abs(base))


@pytest.mark.parametrize("params", [
    base_params(contexts=[1], heads=2, num_kv_splits=8),
    base_params(contexts=[17], heads=8, num_kv_splits=2),
    base_params(contexts=[65], heads=24, num_kv_splits=3),
    base_params(contexts=[17], latent_dim=256, rope_dim=32, num_kv_splits=2),
], ids=["empty-splits", "eight-heads", "two-head-blocks", "288-row-cache"])
@pytest.mark.parametrize("copy_engines", [False, True])
def test_non_campaign_geometries_conserve_the_per_kernel_ledger(params, copy_engines):
    op = mla_decode.make_op(params)
    report = verify_operator(
        hardware_case(copy_engines=copy_engines), op, mla_decode.build,
        kernels=mla_decode.kernels,
    )
    assert report["status"] == "passed", report
    assert report["work_mismatches"] == {}
    assert len(report["result"]["kernels"]) == 2
    assert report["buffers_released"] and report["finite_completion"]


# ---------------------------------------------------------------------------
# Two explicit launch children, host submissions and metadata non-effects.
# ---------------------------------------------------------------------------


def run_execution(op, *, recipe=None, replay_scope="none", units=4,
                  launch_us=50):
    case = parse_case(hardware_case(units=units, launch_us=launch_us))
    recipes = recipe or RecipeBook(recipes={
        mla_decode.IMPLEMENTATION: PlannedRecipe(
            mla_decode.build, mla_decode.kernels
        ),
    })
    execution = Execution(case, CostModel(case), recipes=recipes,
                          trace_limit=4096)
    plan = IterationPlan((RequestView("r", 1, 0, True, "decode"),))
    iteration = IterationWork((op,), plan, 1, 1, 1, None, "decode", "one",
                              replay_scope=replay_scope)
    result = execution.add(iteration, 0, 0)
    execution.engine.advance(until=(result.completion,))
    return execution, result


def host_submit_tasks(execution):
    return [task for task in execution.engine.trace
            if task["task"].endswith(":submit")]


def test_eager_execution_submits_both_source_kernels_and_capture_collapses():
    op = mla_decode.make_op(base_params(contexts=[17], num_kv_splits=2))
    eager, result = run_execution(op)
    submits = host_submit_tasks(eager)
    assert len(submits) == 2
    assert eager.engine.serviced["host@r0:s0"] == pytest.approx(2 * 0.05)
    assert result.operators == 3  # two kernels plus the terminal join barrier
    # Both device children executed real work.
    assert eager.engine.serviced["matrix@r0:s0"] > 0
    assert eager.engine.serviced["vector@r0:s0"] > 0
    assert eager.engine.serviced["memory@r0:s0"] > 0

    captured, _ = run_execution(op, replay_scope="full")
    assert len(host_submit_tasks(captured)) == 1
    assert captured.engine.serviced["host@r0:s0"] == pytest.approx(0.05)
    assert captured.engine.summary()["live_tasks"] == 0
    assert captured.engine.summary()["live_programs"] == 0


def test_source_call_cannot_silently_drop_the_second_kernel():
    op = mla_decode.make_op(base_params(contexts=[17], num_kv_splits=2))
    assert dict(op.metadata)["source_kernel_launches"] == 2

    def stage1_only(parent, geometry):
        stage1 = mla_decode.kernels(parent)[0]
        return mla_decode.build(stage1, geometry)

    # A hand-written planner that keeps only stage1 cannot bypass the call's
    # physical launch boundaries.
    single = RecipeBook(recipes={
        mla_decode.IMPLEMENTATION: PlannedRecipe(stage1_only),
    })
    with pytest.raises(ValueError, match="source_kernel_expansion_required"):
        run_execution(op, recipe=single)

    case = parse_case(hardware_case(units=4, launch_us=50))
    expanded = operator_costs(
        case, (op,), recipes=RecipeBook(recipes={
            mla_decode.IMPLEMENTATION: PlannedRecipe(
                mla_decode.build, mla_decode.kernels
            ),
        })
    )
    assert expanded["operator_occurrences"] == 2
    with pytest.raises(ValueError, match="source_kernel_expansion_required"):
        operator_costs(case, (op,), recipes=single)


def test_kernel_children_have_distinct_ids_dependencies_and_phases():
    op = mla_decode.make_op(base_params(contexts=[17], num_kv_splits=2))
    children = mla_decode.kernels(op)
    assert len(children) == 2
    first, second = children
    assert (first.op_id, second.op_id) == (
        "stage1_grouped_decode", "stage2_split_merge"
    )
    assert first.deps == ()
    assert second.deps == ("stage1_grouped_decode",)
    assert dict(first.metadata)["kernel_phase"] == "stage1"
    assert dict(second.metadata)["kernel_phase"] == "stage2"
    assert first.launch_shape == (1, 1, 2)
    assert second.launch_shape == (1, 16)
    assert mla_decode.kernels(op) == children  # deterministic, no mutation


def test_build_requires_explicit_kernel_phase_selection():
    op = mla_decode.make_op(base_params())
    with pytest.raises(ValueError, match="kernel_phase_required"):
        mla_decode.build(op, geometry())


# ---------------------------------------------------------------------------
# Physical ownership / per-SM-rate counterexamples.
# ---------------------------------------------------------------------------


def per_sm_hardware(units, *, per_sm_peak=1e12):
    # Fixed per-SM rates: every declared device rate grows with physical units.
    return hardware_case(units=units, bf16_peak=per_sm_peak * units,
                         rate_scale=units, copy_engines=False)


def test_single_cta_does_not_speed_up_when_idle_physical_owners_are_added():
    params = base_params(contexts=[1024], num_kv_splits=1)
    stage1 = mla_decode.kernels(mla_decode.make_op(params))[0]
    rows = {}
    for units in (1, 32):
        report = verify_operator(per_sm_hardware(units), stage1,
                                 mla_decode.build)
        assert report["status"] == "passed", report
        rows[units] = report["result"]["duration_ms"]
    matrix_only = 32 * 2 * BLOCK_H * BLOCK_N * (512 + 64 + 512) / 1e12 * 1000
    # 32 key tiles, one CTA, one SM's fixed rate: no 32x speedup from idle SMs.
    # The single CTA also serialises its Q/K loads and vector tail, so its
    # duration is bounded below by the matrix work and above by matrix plus
    # all of its declared memory/vector demand.
    # Shared HBM bandwidth may scale with the device, so allow a few percent;
    # the matrix work of the one CTA is still served by one physical owner.
    assert rows[32] == pytest.approx(rows[1], rel=0.05)
    assert rows[32] >= 0.99 * matrix_only
    assert rows[32] < 1.1 * matrix_only
    assert rows[32] > rows[1] / 10  # the rejected 32x-envelope counterexample


def test_thirty_two_ctas_do_use_thirty_two_owners():
    params = base_params(contexts=[1024] * 32, num_kv_splits=1)
    stage1 = mla_decode.kernels(mla_decode.make_op(params))[0]
    one = verify_operator(per_sm_hardware(1), stage1, mla_decode.build)
    many = verify_operator(per_sm_hardware(32), stage1, mla_decode.build)
    assert one["status"] == many["status"] == "passed"
    # 32 independent source CTAs legitimately fill 32 SMs.
    assert one["result"]["duration_ms"] == pytest.approx(
        32 * many["result"]["duration_ms"], rel=1e-9
    )


# ---------------------------------------------------------------------------
# Trace-level CTA ownership, recurrence ordering and per-CTA ledger oracles.
# ---------------------------------------------------------------------------

_CTA_GROUP = re.compile(r":group(\d+):")


def run_child_trace(child, *, units, bf16_peak=1e14, trace_limit=200000,
                    vector_units=None):
    vector_units = vector_units or 2 * units
    plan = mla_decode.build(
        child, Geometry(units, {"vector": vector_units}, {}, {})
    )
    context = RecipeContext(
        child, {}, dict(NAMES), {value: 1.0 for value in NAMES.values()},
        units, compute_peaks={"bf16": bf16_peak},
        resource_units={"cube": units, "vector": vector_units,
                        "mte2": units, "mte3": units},
    )
    engine = Engine([
        Resource(NAMES["cube"], 1.0, units=units),
        Resource(NAMES["vector"], 1e12, units=vector_units),
        Resource(NAMES["hbm"], 1e12, units=units),
        Resource(NAMES["mte2"], 1e12, units=units),
        Resource(NAMES["mte3"], 1e12, units=units),
    ], trace_limit=trace_limit)
    end = plan.lower(context).start(engine, "op", engine.submit("ready"))
    engine.advance(until=(end,))
    return plan, context, engine


def demand_totals(engine):
    totals = {}
    for task in engine.trace:
        for resource, amount in task["demand"].items():
            totals[resource] = totals.get(resource, 0.0) + amount
    return totals


def grouped_tasks(engine, *, with_demand=True):
    groups = {}
    for task in engine.trace:
        match = _CTA_GROUP.search(task["task"])
        if not match:
            continue
        if with_demand and not task["demand"]:
            continue
        groups.setdefault(int(match.group(1)), []).append(task)
    return groups


def test_stage1_ctas_are_pinned_and_execute_qk_softmax_pv_in_order():
    params = base_params(contexts=[129], num_kv_splits=2)
    stage1 = mla_decode.kernels(mla_decode.make_op(params))[0]
    plan, context, engine = run_child_trace(stage1, units=2)
    assert demand_totals(engine) == pytest.approx(context.bind(plan.work),
                                                  rel=1e-12, abs=1e-12)

    groups = grouped_tasks(engine)
    assert set(groups) == {0, 1}
    for core, tasks in groups.items():
        for task in tasks:
            placement = task["placement"]
            assert placement[NAMES["cube"]] == (core,)
            assert placement[NAMES["vector"]] == (2 * core, 2 * core + 1)
            assert placement[NAMES["mte2"]] == (core,)
            assert placement[NAMES["mte3"]] == (core,)

        q_load = [task for task in tasks if task["task"].endswith(":q_load")]
        first_k = [task for task in tasks
                   if task["task"].endswith(":tile0:k_load")]
        assert len(q_load) == len(first_k) == 1
        assert q_load[0]["end_ms"] <= first_k[0]["start_ms"] + 1e-15
        qk = [task for task in tasks if task["task"].endswith(":qk")]
        softmax = [task for task in tasks if task["task"].endswith(":softmax")]
        pv = [task for task in tasks if task["task"].endswith(":pv")]
        # Split 0 has 3 tiles and split 1 has 2 in this 129-key case.
        chunk, spans, _ = split_partition(129, 2)
        span = spans[core]
        expected_tiles = -(-(span[1] - span[0]) // BLOCK_N)
        assert (len(qk), len(softmax), len(pv)) == (
            expected_tiles, expected_tiles, expected_tiles
        )
        for index in range(len(qk)):
            assert qk[index]["end_ms"] <= softmax[index]["start_ms"] + 1e-15
            assert softmax[index]["end_ms"] <= pv[index]["start_ms"] + 1e-15
            if index:
                # Strict per-CTA source order: tile i+1 does not start its
                # QK before tile i's PV has completed.
                assert pv[index - 1]["end_ms"] <= qk[index]["start_ms"] + 1e-15
        assert pv[-1]["end_ms"] <= max(
            task["start_ms"] for task in tasks
            if task["task"].endswith(":scratch_store")
        ) + 1e-15
    assert engine.summary()["peak_live_tasks"] < 100


def test_stage1_per_cta_traffic_matches_independent_source_oracle():
    params = base_params(contexts=[17], num_kv_splits=2)
    stage1 = mla_decode.kernels(mla_decode.make_op(params))[0]
    plan, context, engine = run_child_trace(stage1, units=2)
    groups = grouped_tasks(engine)
    for core, tasks in groups.items():
        expected = oracle_stage1_cta(params, 0, 0, core)
        memory = sum(task["demand"].get(NAMES["hbm"], 0) for task in tasks)
        vector = sum(task["demand"].get(NAMES["vector"], 0) for task in tasks)
        matrix = sum(task["demand"].get(NAMES["cube"], 0) for task in tasks)
        assert memory == expected["q_bytes"] + sum(expected["tile_loads"]) + (
            expected["store_bytes"]
        )
        assert vector == expected["vector"]
        assert matrix == pytest.approx(
            1000 * expected["flops"] / 1e14, rel=1e-12
        )
        # Latent K is read once per key; no second 512-column V read.
        assert memory - expected["q_bytes"] - expected["store_bytes"] == sum(
            expected["tile_loads"]
        )
        _, spans, _ = split_partition(17, 2)
        start, end = spans[core]
        for index, load in enumerate(expected["tile_loads"]):
            valid_n = min(BLOCK_N, end - (start + index * BLOCK_N))
            assert load == (
                valid_n * (INDEX_BYTES + 576 * PROGRAM_BYTES)
                + (2 * ACCUM_BYTES if index == 0 else 0)
            )


def test_stage2_merge_ctas_own_one_head_and_skip_empty_splits():
    params = base_params(contexts=[3], heads=4, num_kv_splits=8)
    stage2 = mla_decode.kernels(mla_decode.make_op(params))[1]
    plan, context, engine = run_child_trace(stage2, units=2)
    assert demand_totals(engine) == pytest.approx(context.bind(plan.work),
                                                  rel=1e-12, abs=1e-12)
    groups = grouped_tasks(engine)
    # Four batch/head CTAs; two per physical owner, serial per owner stream.
    assert set(groups) == {0, 1}
    ctas = 0
    for core, tasks in groups.items():
        for task in tasks:
            assert task["placement"][NAMES["vector"]] == (
                2 * core, 2 * core + 1
            )
        outer = {}
        for task in tasks:
            match = re.search(r":tile\d+:cta:", task["task"])
            if match:
                outer.setdefault(task["task"][:match.end()], []).append(task)
        # Two batch/head CTAs per physical owner in this 4-CTA case.
        assert len(outer) == 2
        for subtasks in outer.values():
            ctas += 1
            loads = [task for task in subtasks
                     if task["task"].endswith(":partial_load")]
            merges = [task for task in subtasks
                      if task["task"].endswith(":online_merge")]
            assert len(loads) == len(merges) == 3  # only the 3 non-empty slots
            assert all(task["demand"][NAMES["hbm"]]
                       == 512 * ACCUM_BYTES + ACCUM_BYTES for task in loads)
            for index in range(3):
                assert loads[index]["end_ms"] <= merges[index]["start_ms"] + 1e-15
                if index:
                    assert merges[index - 1]["end_ms"] <= merges[index][
                        "start_ms"] + 1e-15
            store = [task for task in subtasks
                     if task["task"].endswith(":store")]
            assert len(store) == 1
            assert store[0]["demand"][NAMES["hbm"]] == (
                512 * PROGRAM_BYTES + PROGRAM_BYTES
            )
    assert ctas == len(params["contexts"]) * params["heads"]
    assert engine.summary()["peak_live_tasks"] < 100


def test_empty_split_stage1_ctas_only_load_q_and_sequence_length():
    params = base_params(contexts=[1], heads=2, num_kv_splits=8)
    stage1 = mla_decode.kernels(mla_decode.make_op(params))[0]
    plan, context, engine = run_child_trace(stage1, units=8)
    groups = grouped_tasks(engine)
    assert set(groups) == set(range(8))
    for core, tasks in groups.items():
        stages = [task["task"].split(":")[-1] for task in tasks]
        if core == 0:
            assert stages == [
                "q_load", "k_load", "qk", "softmax", "pv", "scratch_store"
            ]
        else:
            assert stages == ["q_load"]
    assert demand_totals(engine) == pytest.approx(context.bind(plan.work),
                                                  rel=1e-12, abs=1e-12)
    oracle = oracle_work(params)
    assert work_ledger(plan) == oracle["stage1"]


# ---------------------------------------------------------------------------
# Address, tail and page coverage.
# ---------------------------------------------------------------------------


def test_page_and_tail_address_oracle_never_reads_past_the_split_end():
    page_size, length = 16, 17
    page_ids = [token // page_size for token in range(length)]
    assert max(page_ids) + 1 == math.ceil(length / page_size)
    chunk, spans, tiles = split_partition(length, 2)
    covered = []
    for start, end in spans:
        valid = list(range(start, end))
        assert all(0 <= page_ids[token] < math.ceil(length / page_size)
                   for token in valid)
        assert valid[-1] < end
        covered.extend(valid)
    assert covered == list(range(length))
    assert spans[1][0] // page_size != (spans[1][1] - 1) // page_size

    params = base_params(contexts=[length], num_kv_splits=2)
    op = mla_decode.make_op(params)
    plan_meta = dict(op.metadata)["plan"]
    assert plan_meta["pages_per_request"] == (math.ceil(length / page_size),)
    assert plan_meta["page_index_entries"] == length
    assert plan_meta["block_tiles"] == (tiles,)
    for core in range(2):
        expected = oracle_stage1_cta(params, 0, 0, core)
        assert expected["tiles"] == len(
            range(spans[core][0], spans[core][1], BLOCK_N)
        )


# ---------------------------------------------------------------------------
# Laziness, boundedness and head-padding visibility.
# ---------------------------------------------------------------------------


def test_program_is_lazy_and_bounded_for_a_million_token_context():
    params = base_params(contexts=[1_000_000], heads=16, num_kv_splits=8)
    stage1 = mla_decode.kernels(mla_decode.make_op(params))[0]
    plan = mla_decode.build(stage1, geometry(64))
    context = RecipeContext(
        stage1, {}, dict(NAMES), {value: 1.0 for value in NAMES.values()},
        64, compute_peaks={"bf16": 1e14},
    )
    program = plan.program(context)
    assert isinstance(program, ParallelProgram)
    assert len(program.programs) == 8  # one source CTA per (request, split)
    assert all(isinstance(stream, PipelineProgram) for stream in program.programs)

    # Materialise only the first CTA of one stream: its key-tile loop is a
    # bounded pipeline, not an enumerated list of 3907 tile stages.
    cta_stage = program.programs[0].tile(0)[0]
    assert cta_stage.name == "cta"
    assert isinstance(cta_stage.program, KernelProgram)
    key_tiles = [
        stage for stage in cta_stage.program.stages
        if stage.name == "key_tiles"
    ][0]
    assert isinstance(key_tiles.program, PipelineProgram)
    assert key_tiles.program.count == -(-(-(-1_000_000 // 8)) // BLOCK_N)
    assert plan.work.matrix["bf16"] > 1e9

    twice = mla_decode.build(
        mla_decode.kernels(mla_decode.make_op(
            base_params(contexts=[2_000_000], heads=16, num_kv_splits=8)
        ))[0],
        geometry(64),
    )
    assert twice.work.matrix["bf16"] == pytest.approx(
        2 * plan.work.matrix["bf16"], rel=1e-3
    )

    # A finite trace on a 32k-token context stays inside the bounded window.
    trace_params = base_params(contexts=[32768], heads=16, num_kv_splits=8)
    trace_stage1 = mla_decode.kernels(mla_decode.make_op(trace_params))[0]
    _, _, engine = run_child_trace(trace_stage1, units=8)
    summary = engine.summary()
    assert summary["live_tasks"] == 0
    assert summary["peak_live_tasks"] < 100
    assert summary["completed_tasks"] > 1000


def test_large_split_grid_is_a_bounded_stream_set_not_an_eager_cta_list():
    params = base_params(contexts=[1], heads=16, num_kv_splits=4096)
    stage1 = mla_decode.kernels(mla_decode.make_op(params))[0]
    plan = mla_decode.build(stage1, geometry(64))
    context = RecipeContext(
        stage1, {}, dict(NAMES), {value: 1.0 for value in NAMES.values()},
        64, compute_peaks={"bf16": 1e14},
    )
    program = plan.program(context)
    assert isinstance(program, ParallelProgram)
    # 4096 source CTAs are represented by 64 one-owner streams, not 4096
    # eagerly built CTA programs.
    assert len(program.programs) == 64
    assert all(stream.count == 64 for stream in program.programs)
    assert dict(mla_decode.make_op(params).metadata)["plan"][
        "stage1_programs"] == 4096


def test_second_source_special_case_uses_its_own_288_row_blocks():
    params = base_params(
        contexts=[17], latent_dim=256, rope_dim=32, num_kv_splits=2
    )
    op = mla_decode.make_op(params)
    dispatch = dict(op.metadata)["dispatch"]
    assert (dispatch["block_dmodel"], dispatch["block_dpe"], dispatch["block_dv"]) == (
        256, 32, 256
    )
    _, _, plan1, plan2 = child_plans(op)
    oracle = oracle_work(params)
    assert work_ledger(plan1) == oracle["stage1"]
    assert work_ledger(plan2) == oracle["stage2"]


def test_minimal_widths_keep_source_tail_rows_visible():
    params = base_params(contexts=[1], heads=8, num_kv_splits=1)
    op = mla_decode.make_op(params)
    plan1 = mla_decode.build(mla_decode.kernels(op)[0], geometry())
    assert work_ledger(plan1)["flops"] == 1114112
    assert dict(op.metadata)["plan"]["padded_heads"] == 16

    params = base_params(contexts=[1], heads=24, num_kv_splits=1)
    op = mla_decode.make_op(params)
    plan1 = mla_decode.build(mla_decode.kernels(op)[0], geometry())
    assert work_ledger(plan1)["flops"] == 2 * 1114112
    assert dict(op.metadata)["plan"]["head_blocks"] == 2
    assert dict(op.metadata)["plan"]["padded_heads"] == 32


# ---------------------------------------------------------------------------
# Unsupported neighbouring paths must be reported, never substituted.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"implementation": "triton_decode_attention"}, "unsupported_implementation"),
        ({"is_mla": False}, "unsupported_dispatch:is_mla"),
        ({"is_hip": True}, "unsupported_dispatch:is_hip"),
        ({"logit_cap": 0.25}, "unsupported_dispatch:logit_cap"),
        ({"cache_dtype": "fp8"}, "unsupported_dtype:cache_dtype"),
        ({"dtype": "fp16"}, "unsupported_dtype:dtype"),
        ({"key_heads": 2}, "unsupported_dispatch:key_heads"),
        ({"heads": 1}, "unsupported_dispatch:heads"),
        ({"page_size": 8}, "unsupported_page_size"),
        ({"latent_dim": 500, "rope_dim": 76}, "unsupported_head_dim"),
        ({"contexts": []}, "contexts_must_be_non_empty"),
        ({"contexts": [0]}, "integer_out_of_range"),
        ({"num_kv_splits": 0}, "integer_out_of_range"),
    ],
)
def test_unsupported_paths_raise_instead_of_selecting_a_neighbour(changes, match):
    with pytest.raises(ValueError, match=match):
        mla_decode.make_op(base_params(**changes))


def test_missing_explicit_split_count_is_reported():
    params = base_params()
    del params["num_kv_splits"]
    with pytest.raises(ValueError, match="parameter_required:num_kv_splits"):
        mla_decode.make_op(params)


def test_invalid_geometry_is_rejected_before_execution():
    op = mla_decode.make_op(base_params())
    with pytest.raises(ValueError, match="positive_compute_units"):
        mla_decode.build(op, Geometry(0, {}, {}))
    with pytest.raises(ValueError, match="resource_units_must_be_positive"):
        mla_decode.build(op, Geometry(2, {"vector": 0}, {}))
