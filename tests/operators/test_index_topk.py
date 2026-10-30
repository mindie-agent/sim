"""Independent arithmetic, address, ownership and launch oracles for index_topk.

[Role] Check the source-derived plan against NumPy references and actual
  engine service, not against strings or the planner's own helper functions.
[Invariants] The fixed campaign cases keep their declared dispatch; padded
  score/sort work, mask lanes, selected-index bytes, per-CTA ownership and the
  two launch boundaries are independently recomputed.  Synthetic hardware rates
  are structural controls, never accuracy evidence.
[Timeouts & Locks] Small NumPy problems and bounded virtual-engine runs; the
  large-geometry check never materializes one event per token.
[Side-Effect Policy] No device runtime, measurements, files or network.
"""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operators import index_topk as subject
from mindie_sim.prediction.program import (
    KernelProgram, ParallelProgram, PipelineProgram, RecipeBook, RecipeContext,
)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView

ROOT = Path(__file__).resolve().parents[2]
SPARSE = 128
SCORE_Q_TILE = 64


# ---------------------------------------------------------------------------
# Independent structural arithmetic.  This deliberately re-derives the work
# with per-tile and per-token loops rather than calling the planner helpers.
# ---------------------------------------------------------------------------
def _base_params(**overrides):
    params = {
        "query": 3,
        "context": 640,
        "head_dim": 128,
        "topk": 2,
        "block_size_q": 1,
        "score_block": 128,
        "topk_block": 256,
        "init_blocks": 0,
        "local_blocks": 1,
        "dtype": "bf16",
        "implementation": "triton_block_index_topk",
        "mask_init": False,
        "mask_local": False,
    }
    params.update(overrides)
    return params


def _ceil_div(value, divisor):
    return -((-value) // divisor)


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _expected_counts(params):
    query = params["query"]
    context = params["context"]
    batch = params.get("batch", 1)
    heads = params.get("heads", 1)
    head_dim = params["head_dim"]
    topk = params["topk"]
    tile = params["topk_block"]
    dtype_width = 2  # bf16
    sequence = context + query
    max_blocks = _ceil_div(sequence, SPARSE)
    score_tiles = _ceil_div(query, SCORE_Q_TILE)
    stride = _align_up(max_blocks, 16)
    block_size_t = 1 << (topk - 1).bit_length()

    qk_flops = q_bytes = k_bytes = table_bytes = 0
    score_store = mask_lanes = max_lanes = addr_lanes = 0
    for index in range(score_tiles):
        rows = min(SCORE_Q_TILE, query - index * SCORE_Q_TILE)
        q_start = context + index * SCORE_Q_TILE
        hi = min(sequence, q_start + SCORE_Q_TILE)
        blocks = _ceil_div(hi, SPARSE)
        masked = sum(1 for block in range(blocks)
                     if q_start < block * SPARSE + SPARSE)
        qk_flops += 2 * SCORE_Q_TILE * SPARSE * head_dim * blocks
        q_bytes += rows * head_dim * dtype_width
        k_bytes += blocks * SPARSE * head_dim * dtype_width
        table_bytes += blocks * 4
        score_store += rows * blocks * 4
        mask_lanes += SCORE_Q_TILE * SPARSE * masked
        max_lanes += SCORE_Q_TILE * (SPARSE - 1) * blocks
        addr_lanes += SPARSE * blocks + SCORE_Q_TILE

    qk_flops *= batch * heads
    q_bytes *= batch * heads
    k_bytes *= batch * heads
    table_bytes *= batch * heads
    score_store *= batch * heads
    mask_lanes *= batch * heads
    max_lanes *= batch * heads
    addr_lanes *= batch * heads

    iterations = extra = padded = cas_lanes = merge_lanes = 0
    visible = selected = init_candidates = local_candidates = 0
    log2_tile = tile.bit_length() - 1
    first_stages = log2_tile * (log2_tile + 1) // 2
    later_stages = log2_tile * (log2_tile - 1) // 2 + 2 * log2_tile
    for _req in range(batch):
        for _head in range(heads):
            for token in range(query):
                v = (context + token) // SPARSE + 1
                count = _ceil_div(v, tile)
                iterations += count
                extra += count - 1
                padded += count * tile
                cas_lanes += tile * (first_stages + (count - 1) * later_stages)
                merge_lanes += tile * (count - 1)
                visible += v
                selected += min(topk, v)
                init_candidates += min(params.get("init_blocks", 0), v)
                local_candidates += min(params.get("local_blocks", 0), v)

    prepare_ops = 15 * padded
    sort_ops = 6 * cas_lanes
    merge_ops = 6 * merge_lanes
    init_ops = 2 * tile * query * batch * heads
    extract_ops = (3 * tile + block_size_t) * query * batch * heads
    score_vector_ops = 2 * mask_lanes + max_lanes + addr_lanes
    vector_ops = (
        score_vector_ops + prepare_ops + sort_ops + merge_ops
        + init_ops + extract_ops
    )
    output_bytes = query * batch * heads * topk * 4
    score_read_bytes = visible * 4
    reads = q_bytes + k_bytes + table_bytes + score_read_bytes
    return {
        "score_tiles": score_tiles,
        "max_blocks": max_blocks,
        "score_stride": stride,
        "block_size_t": block_size_t,
        "score_programs": batch * score_tiles * heads,
        "topk_programs": batch * query * heads,
        "score_qk_flops": qk_flops,
        "score_q_bytes": q_bytes,
        "score_k_bytes": k_bytes,
        "score_block_table_bytes": table_bytes,
        "score_store_bytes": score_store,
        "score_read_bytes": score_read_bytes,
        "score_mask_lanes": mask_lanes,
        "score_max_lanes": max_lanes,
        "score_addr_lanes": addr_lanes,
        "score_vector_ops": score_vector_ops,
        "topk_iterations": iterations,
        "topk_extra_iterations": extra,
        "topk_padded_lanes": padded,
        "topk_sort_stages": cas_lanes // tile,
        "topk_cas_lanes": cas_lanes,
        "topk_merge_lanes": merge_lanes,
        "topk_first_iteration_stages": first_stages,
        "topk_later_iteration_stages": later_stages,
        "topk_prepare_ops": prepare_ops,
        "topk_sort_ops": sort_ops,
        "topk_merge_ops": merge_ops,
        "topk_init_ops": init_ops,
        "topk_extract_ops": extract_ops,
        "vector_ops": vector_ops,
        "score_causal_mask_lanes": mask_lanes,
        "topk_causal_mask_lanes": padded,
        "topk_init_mask_lanes": padded,
        "topk_local_mask_lanes": padded,
        "visible_candidates": visible,
        "init_forced_candidates": init_candidates,
        "local_forced_candidates": local_candidates,
        "valid_visible_scores": visible,
        "selected_valid_indices": selected,
        "selected_padded_indices": output_bytes // 4,
        "selected_valid_bytes": selected * 4,
        "selected_index_bytes": output_bytes,
        "score_workspace_bytes": heads * batch * query * stride * 4,
        "hbm_read_bytes": reads,
        "mte2_bytes": reads,
        "mte3_bytes": score_store + output_bytes,
        "onchip_score_tile_bytes": (
            SCORE_Q_TILE * head_dim * dtype_width
            + SPARSE * head_dim * dtype_width
            + SCORE_Q_TILE * SPARSE * 4
            + SCORE_Q_TILE * 4
        ),
        "onchip_sort_bytes_per_program": (7 * tile + block_size_t) * 4,
    }


def _hardware(split_copy=True, units=32, scale_with_units=False):
    """Synthetic control; rates are structural, never calibrated measurements."""
    memory = 1e12
    vector = 1e9
    if scale_with_units:
        memory *= units
        vector *= units
    rates = {"memory": memory, "vector": vector}
    if split_copy:
        rates.update(mte2=memory, mte3=memory)
    return {
        "machine": {
            "name": "synthetic-index-topk-control", "num_devices": 1,
            "device": {
                "memory_bytes": 8 * 1024 ** 3, "compute_units": units,
                "compute_peaks": {"bf16": 1e14 * (units if scale_with_units else 1)},
                "rates": rates, "resource_units": {"vector": 2 * units},
            },
        },
        "model": {
            "name": "unlabeled-index-topk", "layers": 1, "hidden_size": 8,
            "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
            "intermediate_size": 16, "vocab_size": 16,
        },
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    }


def _recipe_context(op, core_count=32):
    names = {
        "cube": "matrix@r0:s0", "hbm": "memory@r0:s0",
        "vector": "vector@r0:s0", "mte2": "mte2@r0:s0", "mte3": "mte3@r0:s0",
    }
    return RecipeContext(
        op=op, work={}, resource_names=names,
        rates={name: 1.0 if name.startswith("matrix") else 1e9
               for name in names.values()},
        core_count=core_count, minimum_ms=0.0, setup_ms=0.0,
        pool_names={}, resource_units={"vector": 2 * core_count},
        compute_peaks={"bf16": 1e14},
    )


# ---------------------------------------------------------------------------
# Address oracle: score workspace, K pages and selected-index output.
# ---------------------------------------------------------------------------
def _score_addresses(params):
    query = params["query"]
    batch = params.get("batch", 1)
    heads = params.get("heads", 1)
    context = params["context"]
    sequence = context + query
    max_blocks = _ceil_div(sequence, SPARSE)
    stride = _align_up(max_blocks, 16)
    writes, reads = set(), set()
    for head in range(heads):
        for req in range(batch):
            for token in range(query):
                index = req * query + token
                row = head * batch * query + index
                tile = token // SCORE_Q_TILE
                q_start = context + tile * SCORE_Q_TILE
                blocks = _ceil_div(min(sequence, q_start + SCORE_Q_TILE), SPARSE)
                for block in range(blocks):
                    writes.add(((row * stride) + block) * 4)
                visible = (context + token) // SPARSE + 1
                for block in range(visible):
                    reads.add(((row * stride) + block) * 4)
    return writes, reads, stride


def _output_addresses(params):
    query = params["query"]
    batch = params.get("batch", 1)
    heads = params.get("heads", 1)
    topk = params["topk"]
    addresses = set()
    for head in range(heads):
        for token in range(batch * query):
            row = head * batch * query + token
            for slot in range(topk):
                addresses.add((row * topk + slot) * 4)
    return addresses


# ---------------------------------------------------------------------------
# Dense block-score and source-bitonic selection oracles (NumPy only).
# ---------------------------------------------------------------------------
def _dense_block_scores(q, pages, table, context, init_blocks, local_blocks,
                        mask_init, mask_local):
    query, head_dim = q.shape
    blocks = pages.shape[0]
    keys = pages[table].reshape(blocks * SPARSE, head_dim).astype(np.float64)
    raw = q.astype(np.float64) @ keys.T
    q_pos = context + np.arange(query)
    k_pos = np.arange(blocks * SPARSE)
    raw = np.where(k_pos[None, :] <= q_pos[:, None], raw, -np.inf)
    scores = raw.reshape(query, blocks, SPARSE).max(axis=2)
    visible = (q_pos + SPARSE) // SPARSE
    forced = scores.copy()
    for token in range(query):
        count = int(visible[token])
        row = forced[token, :count].copy()
        if init_blocks:
            end = min(init_blocks, count)
            row[:end] = row[:end] - 1e29 if mask_init else 1e30
        if local_blocks:
            start = max(0, count - local_blocks)
            row[start:] = row[start:] - 1e28 if mask_local else 1e29
        forced[token, :count] = row
    return forced, visible


def _dense_topk_indices(scores, visible, topk):
    result = []
    for token in range(scores.shape[0]):
        count = int(visible[token])
        take = min(topk, count)
        order = np.argsort(-scores[token, :count], kind="stable")[:take]
        result.append({int(index) for index in order})
    return result


def _bitonic_merge(values, ids, stage, order, n_dims):
    """Mirror _bitonic_merge/_compare_and_swap, source lines 31-70."""
    count = values.size
    if order == 2:
        flip_shape = (2 ** (n_dims - 1 - stage), 2, 2 ** stage)
        flip = np.broadcast_to(
            np.arange(2)[None, :, None], flip_shape
        ).reshape(-1).astype(bool)
    else:
        flip = np.full(count, bool(order))
    for index in range(stage):
        j = index + (n_dims - stage)
        shape = (2 ** j, 2, 2 ** (n_dims - j - 1))
        y = values.reshape(shape)
        yi = ids.reshape(shape)
        direction = flip.reshape(shape)[:, 0, :]
        cond = (y[:, 0, :] > y[:, 1, :]) != direction
        left = np.where(cond, y[:, 1, :], y[:, 0, :])
        right = np.where(cond, y[:, 0, :], y[:, 1, :])
        values = np.stack((left, right), axis=1).reshape(-1)
        left_i = np.where(cond, yi[:, 1, :], yi[:, 0, :])
        right_i = np.where(cond, yi[:, 0, :], yi[:, 1, :])
        ids = np.stack((left_i, right_i), axis=1).reshape(-1)
    return values, ids


def _source_streaming_topk(scores, topk, tile, block_size_t):
    """Mirror the per-token loop in _topk_index_kernel, lines 226-284."""
    n_dims = tile.bit_length() - 1
    left_half = np.arange(tile) < tile // 2
    topk_score = np.full(tile, -1e30, dtype=np.float64)
    topk_idx = np.zeros(tile, dtype=np.int64)
    off = np.arange(tile)
    for start in range(0, len(scores), tile):
        causal = start + off < len(scores)
        score = np.full(tile, -1e30, dtype=np.float64)
        score[causal] = scores[start:start + tile][: int(causal.sum())]
        last_score = topk_score.copy()
        last_idx = topk_idx.copy()
        topk_score = score
        topk_idx = np.where(causal, start + off + 1, 0)
        for stage in range(1, n_dims):
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx, stage, 2, n_dims
            )
        if start != 0:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx, n_dims, False, n_dims
            )
            merged_score = last_score * left_half + topk_score * (1 - left_half)
            merged_idx = last_idx * left_half + topk_idx * (1 - left_half)
            topk_score, topk_idx = _bitonic_merge(
                merged_score, merged_idx, n_dims, True, n_dims
            )
        else:
            topk_score, topk_idx = _bitonic_merge(
                topk_score, topk_idx, n_dims, True, n_dims
            )
    return [int(index) for index in (topk_idx[:block_size_t] - 1)[:topk]]


# ---------------------------------------------------------------------------
# Program walkers used only on small geometries.
# ---------------------------------------------------------------------------
def _collect_program_work(program):
    total = {}

    def add(work):
        for name, amount in work.items():
            total[name] = total.get(name, 0) + amount

    _collect_program_work_into(program, add)
    return total


def _collect_program_work_into(program, add):
    if isinstance(program, KernelProgram):
        for stage in program.stages:
            add(stage.work)
            if stage.program is not None:
                _collect_program_work_into(stage.program, add)
    elif isinstance(program, ParallelProgram):
        for child in program.programs:
            _collect_program_work_into(child, add)
    elif isinstance(program, PipelineProgram):
        for index in range(program.count):
            for stage in program.tile(index):
                add(stage.work)
                if stage.program is not None:
                    _collect_program_work_into(stage.program, add)
    else:
        raise AssertionError(f"unexpected program type: {type(program).__name__}")


def _pipeline_programs(program):
    """Yield every (launch name, per-unit PipelineProgram) in a plan program."""
    launches = []
    for stage in program.stages:
        if not isinstance(stage.program, KernelProgram):
            continue
        grid = stage.program.stages[0]
        if not isinstance(grid.program, ParallelProgram):
            continue
        for pipeline in grid.program.programs:
            if not isinstance(pipeline, PipelineProgram):
                raise AssertionError("CTA stream is not a PipelineProgram")
            launches.append((stage.name, pipeline))
    return launches


def _launch_streams(program, launch_name):
    return [pipeline for name, pipeline in _pipeline_programs(program)
            if name == launch_name]


def _stage_tree(stage):
    """Yield a stage and its nested lazy pipeline stages (small geometries only)."""
    yield stage
    if isinstance(stage.program, PipelineProgram):
        for index in range(stage.program.count):
            for nested in stage.program.tile(index):
                yield from _stage_tree(nested)


# ---------------------------------------------------------------------------
# Fixed campaign cases and dispatch.
# ---------------------------------------------------------------------------
def _fixed_cases():
    raw = json.loads(
        (ROOT / "tests/fixtures/operator_cases/index_topk.json").read_text()
    )
    return raw["cases"]


@pytest.mark.parametrize("case", _fixed_cases(), ids=lambda case: case["id"])
def test_fixed_case_dispatch_and_independent_work(case):
    params = case["params"]
    op = subject.make_op(params)
    selected = dict(op.metadata)["dispatch"]
    for key, expected in case["expected_dispatch"].items():
        assert selected[key] == expected
    # The in-tree prefill wrapper passes both constexpr masks false.
    assert selected["mask_init"] is False
    assert selected["mask_local"] is False
    assert selected["mask_branch_selection"] == "wrapper_fixed_false_false"
    assert selected["wrapper_mask_init"] is False
    assert selected["wrapper_mask_local"] is False
    assert selected["init_mask_semantics"] == "force_1e30"
    assert selected["local_mask_semantics"] == "force_1e29"
    assert selected["topk_block"] == params["topk_block"] == 256
    assert selected["score_q_tile"] == 64
    assert selected["block_size_q"] == 1
    assert selected["launch_count"] == 2
    assert selected["kernel_phases"] == ["score", "topk"]
    assert selected["decode_split_k_path"] is False
    assert selected["autotune_configs"] == [64, 128, 256, 512, 1024, 2048]

    counts = dict(op.metadata)["work_components"]
    expected = _expected_counts(params)
    assert counts == expected, {
        key: (counts.get(key), expected.get(key))
        for key in counts.keys() | expected.keys()
        if counts.get(key) != expected.get(key)
    }

    plan = subject.build(op, Geometry(32, {"vector": 64}, {}))
    assert dict(plan.work.matrix) == {"bf16": expected["score_qk_flops"]}
    assert dict(plan.work.resources) == {
        "hbm": expected["hbm_read_bytes"] + expected["mte3_bytes"],
        "mte2": expected["mte2_bytes"],
        "mte3": expected["mte3_bytes"],
        "vector": expected["vector_ops"],
    }
    boundaries = dict(op.metadata)["launch_boundaries"]
    assert boundaries["score"]["kernel"] == "_index_block_score_kernel"
    assert boundaries["topk"]["kernel"] == "_topk_index_kernel"
    assert boundaries["topk"]["after"] == ["score"]
    assert boundaries["score"]["programs"] == counts["score_programs"]
    assert boundaries["topk"]["programs"] == counts["topk_programs"]
    assert boundaries["score"]["dependencies"]["qk_dot"] == ["k_load"]
    assert boundaries["score"]["dependencies"]["mask_max"] == ["qk_dot"]
    assert "load_mma" not in boundaries["score"]["stages"]
    assert boundaries["topk"]["dependencies"]["bitonic_merge_combine"] == [
        "bitonic_merge_desc"]
    cadence = dict(op.metadata)["source_cadence"]
    assert cadence["score_load_dot_separate_stage"] is True
    assert cadence["aggregate_read_matrix_envelope"] is False


@pytest.mark.parametrize("case", _fixed_cases(), ids=lambda case: case["id"])
def test_fixed_case_public_contract_and_actual_service(case):
    params = case["params"]
    expected = _expected_counts(params)
    children = subject.kernels(subject.make_op(params))
    assert [child.op_id for child in children] == ["score", "topk"]
    for split_copy in (False, True):
        op = subject.make_op(params)
        report = verify_operator(
            _hardware(split_copy), op, subject.build, kernels=subject.kernels
        )
        assert report["status"] == "passed"
        assert report["finite_completion"]
        assert report["buffers_released"]
        assert report["work_mismatches"] == {}
        result = report["result"]
        assert len(result["kernels"]) == 2
        phases = {dict(row["operator"]["metadata"])["kernel_phase"]: row
                  for row in result["kernels"]}
        assert set(phases) == {"score", "topk"}
        score, topk = phases["score"]["operator"], phases["topk"]["operator"]
        assert score["flops"] == expected["score_qk_flops"]
        assert dict(score["matrix_flops"]) == {"bf16": expected["score_qk_flops"]}
        assert score["hbm_bytes"] == (
            expected["score_q_bytes"] + expected["score_k_bytes"]
            + expected["score_block_table_bytes"]
            + expected["score_store_bytes"]
        )
        assert score["mte2_bytes"] == (
            expected["score_q_bytes"] + expected["score_k_bytes"]
            + expected["score_block_table_bytes"]
        )
        assert score["mte3_bytes"] == expected["score_store_bytes"]
        assert score["vector_ops"] == expected["score_vector_ops"]
        assert topk["flops"] == 0
        assert topk["hbm_bytes"] == (
            expected["score_read_bytes"] + expected["selected_index_bytes"]
        )
        assert topk["mte2_bytes"] == expected["score_read_bytes"]
        assert topk["mte3_bytes"] == expected["selected_index_bytes"]
        assert topk["vector_ops"] == (
            expected["vector_ops"] - expected["score_vector_ops"]
        )
        declared, serviced = result["declared_service"], result["serviced"]
        assert declared.keys() == serviced.keys()
        for resource, amount in declared.items():
            assert math.isclose(amount, serviced[resource],
                                rel_tol=1e-12, abs_tol=1e-12)
        assert declared["memory@r0:s0"] == (
            expected["hbm_read_bytes"] + expected["mte3_bytes"]
        )
        assert declared["vector@r0:s0"] == expected["vector_ops"]
        assert math.isclose(
            declared["matrix@r0:s0"],
            1000 * expected["score_qk_flops"] / 1e14,
            rel_tol=1e-12,
        )
        if split_copy:
            assert declared["mte2@r0:s0"] == expected["mte2_bytes"]
            assert declared["mte3@r0:s0"] == expected["mte3_bytes"]
        else:
            assert "mte2@r0:s0" not in declared
            assert "mte3@r0:s0" not in declared


def test_build_selects_kernel_phase_from_metadata():
    params = _base_params(query=65, context=1024, topk=16, topk_block=64)
    parent = subject.make_op(params)
    expected = _expected_counts(params)
    score_child, topk_child = subject.kernels(parent)
    score_plan = subject.build(score_child, Geometry(32, {"vector": 64}, {}))
    topk_plan = subject.build(topk_child, Geometry(32, {"vector": 64}, {}))
    assert dict(score_plan.work.matrix) == {"bf16": expected["score_qk_flops"]}
    assert "vector" in score_plan.work.resources
    assert not topk_plan.work.matrix
    assert "vector" in topk_plan.work.resources
    assert dict(topk_plan.work.resources)["hbm"] == (
        expected["score_read_bytes"] + expected["selected_index_bytes"]
    )
    context = _recipe_context(parent)
    score_program = score_plan.program(context)
    topk_program = topk_plan.program(context)
    assert [stage.name for stage in score_program.stages] == ["score_launch"]
    assert [stage.name for stage in topk_program.stages] == ["topk_launch"]
    parent_program = subject.build(parent, Geometry(32, {"vector": 64}, {})).program(context)
    assert [stage.name for stage in parent_program.stages] == [
        "score_launch", "topk_launch",
    ]
    with pytest.raises(ValueError, match="unsupported_index_topk_kernel_phase"):
        tampered = replace(score_child, metadata=tuple(
            (key, "merge" if key == "kernel_phase" else value)
            for key, value in score_child.metadata
        ))
        subject.build(tampered, Geometry(32, {"vector": 64}, {}))


# ---------------------------------------------------------------------------
# Pin every CTA and prove a singleton does not borrow idle per-SM capacity.
# ---------------------------------------------------------------------------
def test_singleton_cta_does_not_speed_up_with_idle_units():
    params = _base_params(query=1, context=512, topk=2)
    op = subject.make_op(params)
    one = verify_operator(_hardware(units=1, scale_with_units=True), op,
                          subject.build, kernels=subject.kernels)
    many = verify_operator(_hardware(units=32, scale_with_units=True), op,
                           subject.build, kernels=subject.kernels)
    assert one["status"] == many["status"] == "passed"
    # Per-SM rates are fixed: one CTA uses one partition at both widths.
    assert many["result"]["duration_ms"] == pytest.approx(
        one["result"]["duration_ms"], rel=1e-12, abs=1e-15)
    # A device-wide aggregate would have made the 32-SM case 32x faster.
    assert many["result"]["duration_ms"] > one["result"]["duration_ms"] / 2

    context = _recipe_context(op)
    program = subject.build(op, Geometry(32, {"vector": 64}, {})).program(context)
    score_streams = _launch_streams(program, "score_launch")
    topk_streams = _launch_streams(program, "topk_launch")
    assert len(score_streams) == 1 and score_streams[0].count == 1
    assert len(topk_streams) == 1 and topk_streams[0].count == 1
    names = context.resource_names
    score_q_load, _score_addr, score_loop = score_streams[0].tile(0)
    assert score_q_load.placement[names["mte2"]] == (0,)
    page_load, k_load, dot, score_mask, score_store = score_loop.program.tile(0)
    assert page_load.placement[names["mte2"]] == (0,)
    assert k_load.placement[names["mte2"]] == (0,)
    assert dot.placement[names["cube"]] == (0,)
    assert len(score_mask.placement[names["vector"]]) == 2
    assert len(score_mask.placement[names["vector"]]) < 64
    assert score_store.placement[names["mte3"]] == (0,)
    init, _chunks, extract, output = topk_streams[0].tile(0)
    topk_load, topk_mask, _scan, _merge = _chunks.program.tile(0)
    assert init.placement[names["vector"]] == score_mask.placement[names["vector"]]
    assert topk_load.placement[names["mte2"]] == (0,)
    assert topk_mask.placement[names["vector"]] == (
        score_mask.placement[names["vector"]]
    )
    assert extract.placement[names["vector"]] == (
        score_mask.placement[names["vector"]]
    )
    assert output.placement[names["mte3"]] == (0,)


def test_every_cta_is_pinned_and_all_grid_ctas_are_covered_lazily():
    params = _base_params(query=65, context=1024, topk=16, topk_block=64,
                          batch=2, heads=2)
    op = subject.make_op(params)
    counts = dict(op.metadata)["work_components"]
    context = _recipe_context(op)
    plan = subject.build(op, Geometry(32, {"vector": 64}, {}))
    program = plan.program(context)
    names = context.resource_names
    for launch, total in (("score_launch", counts["score_programs"]),
                          ("topk_launch", counts["topk_programs"])):
        streams = _launch_streams(program, launch)
        assert len(streams) == min(total, 32)
        assert sum(stream.count for stream in streams) == total
        for stream in streams:
            for index in range(stream.count):
                for stage in stream.tile(index):
                    for leaf in _stage_tree(stage):
                        assert leaf.placement
                        if names["vector"] in leaf.placement:
                            assert len(leaf.placement[names["vector"]]) == 2
                        if names["cube"] in leaf.placement:
                            assert len(leaf.placement[names["cube"]]) == 1
                        if names["mte2"] in leaf.placement:
                            assert len(leaf.placement[names["mte2"]]) == 1
                        if names["mte3"] in leaf.placement:
                            assert len(leaf.placement[names["mte3"]]) == 1
    # Actual per-CTA stage work conserves the plan's aggregate work.
    assert _collect_program_work(program) == pytest.approx(
        context.bind(plan.work), rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# Source instruction cadence: a load and its dependent dot are different
# stages; the top-k merge is carried chunk by chunk.
# ---------------------------------------------------------------------------
def _suffix_tasks(execution, suffix):
    return sorted(
        (event for event in execution.engine.trace
         if event["task"].endswith(":" + suffix)),
        key=lambda event: event["start_ms"],
    )


def _run_trace(op, hardware, trace_limit=5000):
    case = parse_case(hardware)
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(subject.build)})
    execution = Execution(case, CostModel(case), recipes=recipes,
                          trace_limit=trace_limit)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    work = IterationWork((replace(op, deps=()),), plan, 1, 1, 1, None,
                         "prefill", "one")
    result = execution.add(work, 0, 0)
    execution.engine.advance(until=(result.completion,))
    return execution


def _assert_no_simultaneous_read_and_dot(execution):
    mixed = [
        event for event in execution.engine.trace
        if event["demand"].get("matrix@r0:s0")
        and any(event["demand"].get(name) for name in
                ("memory@r0:s0", "mte2@r0:s0", "mte3@r0:s0"))
    ]
    assert mixed == []


def _rate_control(read_bytes, flops):
    """Synthetic rates where the total read and total dot each cost 0.1 ms."""
    hardware = _hardware(split_copy=True, units=1)
    device = hardware["machine"]["device"]
    device["launch_us"] = 0
    device["rates"].update(memory=read_bytes / 1e-4, vector=1e20,
                           mte2=1e20, mte3=1e20)
    device["compute_peaks"]["bf16"] = flops / 1e-4
    return hardware


def test_single_score_block_loads_page_then_dots():
    # Review counterexample: Q=1, C=0, topk=1.  The score read is exactly one
    # q row, one 128x128 bf16 page and one 4-byte block-table entry.
    params = _base_params(query=1, context=0, topk=1)
    score = subject.kernels(subject.make_op(params))[0]
    read = 1 * 128 * 2 + 128 * 128 * 2 + 4
    flops = 2 * SCORE_Q_TILE * SPARSE * 128
    assert read == 33028
    assert flops == 2097152
    plan = subject.build(score, Geometry(1, {"vector": 1, "mte2": 1, "mte3": 1}, {}))
    assert plan.work.resources["mte2"] == read
    assert dict(plan.work.matrix) == {"bf16": flops}

    hardware = _rate_control(read, flops)
    report = verify_operator(hardware, score, subject.build)
    assert report["status"] == "passed"
    # Read 0.1 ms and dot 0.1 ms are dependent; the old single Work envelope
    # charged max(0.1, 0.1) plus negligible store/vector = 0.100012 ms.
    assert report["result"]["duration_ms"] > 0.2
    assert report["result"]["duration_ms"] < 0.201

    execution = _run_trace(score, hardware)
    q_load = _suffix_tasks(execution, "q_load")
    page = _suffix_tasks(execution, "page_load")
    k_load = _suffix_tasks(execution, "k_load")
    dots = _suffix_tasks(execution, "qk_dot")
    assert len(q_load) == len(page) == len(k_load) == len(dots) == 1
    assert q_load[0]["end_ms"] <= page[0]["start_ms"] + 1e-12
    assert page[0]["end_ms"] <= k_load[0]["start_ms"] + 1e-12
    assert k_load[0]["end_ms"] <= dots[0]["start_ms"] + 1e-12
    assert set(dots[0]["demand"]) == {"matrix@r0:s0"}
    _assert_no_simultaneous_read_and_dot(execution)


def test_score_blocks_repeat_the_load_dot_cadence():
    # Five visible pages.  With each total read and total dot priced at 0.1 ms,
    # an aggregate all-read/all-matrix stage would finish in about 0.1 ms; the
    # source loop serialises five page-load/dot pairs.
    params = _base_params(query=1, context=512, topk=2)
    score = subject.kernels(subject.make_op(params))[0]
    blocks = 5
    read = 1 * 128 * 2 + blocks * 128 * 128 * 2 + blocks * 4
    flops = blocks * 2 * SCORE_Q_TILE * SPARSE * 128
    hardware = _rate_control(read, flops)
    report = verify_operator(hardware, score, subject.build)
    assert report["status"] == "passed"
    assert report["result"]["duration_ms"] > 0.2
    assert report["result"]["duration_ms"] < 0.201

    execution = _run_trace(score, hardware)
    k_load = _suffix_tasks(execution, "k_load")
    dots = _suffix_tasks(execution, "qk_dot")
    assert len(k_load) == len(dots) == blocks
    for load, dot in zip(k_load, dots):
        assert load["end_ms"] <= dot["start_ms"] + 1e-12
    for earlier, later in zip(dots, dots[1:]):
        assert earlier["end_ms"] <= later["start_ms"] + 1e-12
    # q_start = 512: blocks 0-3 are fully causal-unmasked; block 4 needs the
    # 2-op compare/select over all 64x128 lanes.
    mask_work = _suffix_tasks(execution, "mask_max")
    assert [event["demand"]["vector@r0:s0"] for event in mask_work] == [
        8256, 8256, 8256, 8256, 24640]
    assert [event["demand"]["memory@r0:s0"] for event in k_load] == [32768] * 5
    _assert_no_simultaneous_read_and_dot(execution)


def test_topk_chunks_merge_sequentially_from_carried_state():
    # 257 visible blocks / 64-wide tile = five sequential chunks.  The source
    # loop's first iteration ends with an ascending merge; later iterations
    # descend, combine the previous top half, then ascend again.
    params = _base_params(query=1, context=32768, topk=16, topk_block=64)
    topk = subject.kernels(subject.make_op(params))[1]
    execution = _run_trace(topk, _hardware(split_copy=True, units=1))
    loads = _suffix_tasks(execution, "load")
    masks = _suffix_tasks(execution, "masks")
    scans = _suffix_tasks(execution, "bitonic_merge_scan")
    desc = _suffix_tasks(execution, "bitonic_merge_desc")
    combines = _suffix_tasks(execution, "bitonic_merge_combine")
    asc = _suffix_tasks(execution, "bitonic_merge_asc")
    extracts = _suffix_tasks(execution, "extract")
    outputs = _suffix_tasks(execution, "output_store")
    assert len(loads) == len(masks) == len(scans) == len(asc) == 5
    assert len(desc) == len(combines) == 4
    assert len(extracts) == len(outputs) == 1
    assert extracts[0]["demand"]["vector@r0:s0"] == (
        64 + (64 - 16) + 2 * 16 + 64)
    assert outputs[0]["demand"]["memory@r0:s0"] == params["topk"] * 4
    assert outputs[0]["start_ms"] >= extracts[0]["end_ms"] - 1e-12
    assert [event["demand"]["memory@r0:s0"] for event in loads] == (
        [64 * 4] * 4 + [4])
    assert [event["demand"]["vector@r0:s0"] for event in masks] == (
        [15 * 64] * 5)

    tile = params["topk_block"]
    log2 = tile.bit_length() - 1
    scan_stages = log2 * (log2 - 1) // 2
    merge_stages = log2
    for index in range(5):
        assert masks[index]["start_ms"] >= loads[index]["end_ms"] - 1e-12
        assert scans[index]["start_ms"] >= masks[index]["end_ms"] - 1e-12
        assert scans[index]["demand"]["vector@r0:s0"] == (
            6 * tile * scan_stages)
        if index == 0:
            assert asc[0]["start_ms"] >= scans[0]["end_ms"] - 1e-12
            continue
        assert desc[index - 1]["start_ms"] >= scans[index]["end_ms"] - 1e-12
        assert combines[index - 1]["start_ms"] >= (
            desc[index - 1]["end_ms"] - 1e-12)
        assert asc[index]["start_ms"] >= (
            combines[index - 1]["end_ms"] - 1e-12)
        assert desc[index - 1]["demand"]["vector@r0:s0"] == (
            6 * tile * merge_stages)
        assert combines[index - 1]["demand"]["vector@r0:s0"] == (
            6 * tile)
        assert asc[index]["start_ms"] >= asc[index - 1]["end_ms"] - 1e-12
        assert loads[index]["start_ms"] >= asc[index - 1]["end_ms"] - 1e-12
    assert loads[-1]["demand"]["memory@r0:s0"] == 4  # 257th score only

    # Independent total over the five chunks and one init/extract pair.
    expected_vector = (
        2 * tile                                  # init fill
        + 5 * 15 * tile                           # per-chunk prepare
        + 5 * 6 * tile * scan_stages              # in-loop compare-exchange
        + 4 * 6 * tile * merge_stages             # descending merges
        + 4 * 6 * tile                            # carried combine
        + 5 * 6 * tile * merge_stages             # ascending merges
        + (tile + (tile - 16) + 2 * 16 + tile)    # extraction and store mask
    )
    assert execution.engine.serviced["vector@r0:s0"] == expected_vector


# ---------------------------------------------------------------------------
# Real host launch boundaries: two eager submissions, one captured group.
# ---------------------------------------------------------------------------
def _host_case(launch_us=50):
    raw = _hardware(split_copy=True, units=4)
    raw["machine"]["device"]["launch_us"] = launch_us
    return parse_case(raw)


def _run_host(scope, *, with_kernels=True, op=None):
    case = _host_case()
    op = op or subject.make_op(_base_params(query=3, context=640, topk=2))
    recipe = PlannedRecipe(subject.build, subject.kernels if with_kernels else None)
    recipes = RecipeBook(recipes={op.recipe: recipe})
    execution = Execution(case, CostModel(case), recipes=recipes, trace_limit=2000)
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    work = IterationWork((op,), plan, 1, 1, 1, None, "prefill", "one",
                         replay_scope=scope)
    result = execution.add(work, 0, 0)
    execution.engine.advance(until=(result.completion,))
    submits = [event for event in execution.engine.trace
               if event["task"].endswith(":submit")]
    return execution, result, submits


def test_two_eager_host_submissions_and_one_captured_group():
    op = subject.make_op(_base_params(query=3, context=640, topk=2))
    expected = {}
    for child_plan in (subject.build(child, Geometry(4, {"vector": 8}, {}))
                       for child in subject.kernels(op)):
        for name, amount in child_plan.work.resources.items():
            expected[name] = expected.get(name, 0) + amount
    eager, eager_result, eager_submits = _run_host("none")
    captured, captured_result, captured_submits = _run_host("full")
    assert eager.engine.serviced["host@r0:s0"] == pytest.approx(0.10)
    assert captured.engine.serviced["host@r0:s0"] == pytest.approx(0.05)
    assert len(eager_submits) == 2
    assert {event["task"].rsplit(":", 1)[0].rsplit("/", 1)[-1]
            for event in eager_submits} == {"score", "topk"}
    assert len(captured_submits) == 1
    # Real device work is the same in both modes; only host submission differs.
    for engine in (eager.engine, captured.engine):
        assert engine.serviced["vector@r0:s0"] == pytest.approx(
            expected["vector"], rel=1e-12)
        assert engine.serviced["memory@r0:s0"] == pytest.approx(
            expected["hbm"], rel=1e-12)
        assert engine.serviced["mte2@r0:s0"] == pytest.approx(
            expected["mte2"], rel=1e-12)
        assert engine.serviced["mte3@r0:s0"] == pytest.approx(
            expected["mte3"], rel=1e-12)
    # Host submission is the only difference; device work and its completion
    # are unchanged, and the host service is counted independently.
    assert eager_result.completion.time == pytest.approx(
        captured_result.completion.time, rel=1e-12, abs=1e-15)


def test_source_call_requires_physical_kernel_expansion_before_host_submissions():
    # A wrapper plan can preserve resource work but still erase a host
    # submission. The public path requires its kernel expansion explicitly.
    op = subject.make_op(_base_params(query=3, context=640, topk=2))
    assert dict(op.metadata)["dispatch"]["launch_count"] == 2
    with pytest.raises(ValueError, match="source_kernel_expansion_required"):
        _run_host("none", with_kernels=False, op=op)


# ---------------------------------------------------------------------------
# Independent identities and raw constexpr branch counterexamples.
# ---------------------------------------------------------------------------
def test_identity_fields_do_not_change_physical_cost():
    params = _base_params()
    op = subject.make_op(params)
    renamed = replace(op, op_id="unrelated", layer=97, requests=("unseen",))
    left = verify_operator(_hardware(), op, subject.build,
                           kernels=subject.kernels)["result"]
    right = verify_operator(_hardware(), renamed, subject.build,
                            kernels=subject.kernels)["result"]
    assert left["duration_ms"] == right["duration_ms"]
    assert left["serviced"] == right["serviced"]


def test_wrapper_false_false_counterexample_selects_forced_local_block():
    # Review counterexample: five visible scores, topk=2, local_blocks=1.
    # Wrapper MASK_LOCAL=False forces the local block to +1e29 and selects
    # block 4; the raw MASK_LOCAL=True branch subtracts 1e28 and selects 0,1.
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    visible = np.array([5])
    wrapper = scores.copy()
    wrapper[4] = 1e29
    raw = scores.copy()
    raw[4] -= 1e28
    assert {4, 0} == _dense_topk_indices(wrapper[None, :], visible, 2)[0]
    assert {0, 1} == _dense_topk_indices(raw[None, :], visible, 2)[0]

    wrapper_op = subject.make_op(_base_params(mask_init=False, mask_local=False))
    assert dict(wrapper_op.metadata)["dispatch"]["local_mask_semantics"] == "force_1e29"
    raw_op = subject.make_op(_base_params(mask_init=False, mask_local=True))
    assert dict(raw_op.metadata)["dispatch"]["local_mask_semantics"] == "mask_minus_1e28"
    assert dict(raw_op.metadata)["dispatch"]["mask_branch_selection"] == "raw_constexpr_branch"
    # The constexpr branches change selection semantics, not declared work.
    assert (dict(wrapper_op.metadata)["work_components"]
            == dict(raw_op.metadata)["work_components"])


@pytest.mark.parametrize(
    "mask_init,mask_local", [(False, False), (False, True),
                             (True, False), (True, True)],
)
def test_raw_constexpr_true_branches_match_dense_source_oracle(mask_init, mask_local):
    params = _base_params(query=3, context=640, topk=2, init_blocks=1,
                          local_blocks=1, mask_init=mask_init,
                          mask_local=mask_local)
    rng = np.random.default_rng(11)
    query, head_dim = params["query"], params["head_dim"]
    blocks = _ceil_div(params["context"] + query, SPARSE)
    q = rng.normal(size=(query, head_dim)).astype(np.float32)
    pages = rng.normal(size=(blocks, SPARSE, head_dim)).astype(np.float32)
    table = rng.permutation(blocks)
    forced, visible = _dense_block_scores(
        q, pages, table, params["context"], params["init_blocks"],
        params["local_blocks"], mask_init, mask_local,
    )
    dense = _dense_topk_indices(forced, visible, params["topk"])
    tile = params["topk_block"]
    block_size_t = 1 << (params["topk"] - 1).bit_length()
    for token in range(query):
        got = _source_streaming_topk(
            forced[token, :visible[token]], params["topk"], tile, block_size_t,
        )
        assert {index for index in got if index >= 0} == dense[token]

    selected_op = subject.make_op(params)
    counts = dict(selected_op.metadata)["work_components"]
    assert counts["topk_iterations"] == query
    assert counts["topk_padded_lanes"] == query * tile
    assert counts["selected_valid_indices"] == sum(
        min(params["topk"], int(v)) for v in visible
    )
    assert counts["selected_padded_indices"] == query * params["topk"]
    dispatch = dict(selected_op.metadata)["dispatch"]
    assert dispatch["mask_init"] is mask_init
    assert dispatch["mask_local"] is mask_local
    assert dispatch["mask_branch_selection"] == (
        "raw_constexpr_branch" if (mask_init or mask_local)
        else "wrapper_fixed_false_false"
    )


# ---------------------------------------------------------------------------
# Padded work, tails, layout and bytes.
# ---------------------------------------------------------------------------
def test_dense_scores_match_streaming_bitonic_across_iterations():
    params = _base_params(query=1, context=8192, topk=16, topk_block=64,
                          init_blocks=1, local_blocks=1)
    rng = np.random.default_rng(23)
    blocks = _ceil_div(params["context"] + params["query"], SPARSE)
    q = rng.normal(size=(1, params["head_dim"])).astype(np.float32)
    pages = rng.normal(size=(blocks, SPARSE, params["head_dim"])).astype(np.float32)
    table = rng.permutation(blocks)
    forced, visible = _dense_block_scores(
        q, pages, table, params["context"], 1, 1, False, False,
    )
    assert visible[0] == blocks == 65
    dense = _dense_topk_indices(forced, visible, params["topk"])[0]
    got = _source_streaming_topk(
        forced[0, :visible[0]], params["topk"], 64, 16,
    )
    assert {index for index in got if index >= 0} == dense
    counts = dict(subject.make_op(params).metadata)["work_components"]
    assert counts["topk_iterations"] == 2
    assert counts["topk_padded_lanes"] == 128
    assert counts["topk_sort_stages"] == 21 + 27


def test_streaming_bitonic_keeps_one_real_index_and_pads_with_minus_one():
    # One visible block, top-k 4: the source extracts the first 4 sorted slots
    # and writes -1 for every slot beyond valid_blocks (lines 281-284).
    got = _source_streaming_topk(np.array([0.5]), 4, 64, 4)
    assert got[0] == 0
    assert got[1:] == [-1, -1, -1]


@pytest.mark.parametrize(
    "overrides",
    [
        {"query": 128, "context": 100, "topk": 8, "topk_block": 64},
        {"query": 7, "context": 129, "topk": 4, "topk_block": 64},
        {"query": 1, "context": 0, "topk": 4, "topk_block": 64},
        {"query": 65, "context": 8192, "topk": 16, "topk_block": 64},
        {"query": 200, "context": 1000, "topk": 8, "topk_block": 128,
         "batch": 2, "heads": 2, "init_blocks": 3, "local_blocks": 4},
    ],
)
def test_closed_form_matches_token_enumeration(overrides):
    params = _base_params(**overrides)
    counts = dict(subject.make_op(params).metadata)["work_components"]
    expected = _expected_counts(params)
    assert counts == expected, {
        key: (counts.get(key), expected.get(key))
        for key in counts.keys() | expected.keys()
        if counts.get(key) != expected.get(key)
    }


def test_forced_block_counts_clamp_to_visible_candidates():
    params = _base_params(query=17, context=301, topk=4, topk_block=64,
                          init_blocks=10 ** 6, local_blocks=10 ** 6)
    counts = dict(subject.make_op(params).metadata)["work_components"]
    assert counts["init_forced_candidates"] == counts["visible_candidates"]
    assert counts["local_forced_candidates"] == counts["visible_candidates"]
    assert counts["topk_local_mask_lanes"] == counts["topk_padded_lanes"]
    assert counts["topk_init_mask_lanes"] == counts["topk_padded_lanes"]


def test_padded_sort_and_selected_index_bytes_at_block_boundary():
    params = _base_params(query=1, context=32768, topk=16, topk_block=256)
    op = subject.make_op(params)
    counts = dict(op.metadata)["work_components"]
    assert counts["max_blocks"] == 257
    assert counts["score_stride"] == 272
    assert counts["topk_iterations"] == 2
    assert counts["topk_padded_lanes"] == 512
    assert counts["topk_first_iteration_stages"] == 36
    assert counts["topk_later_iteration_stages"] == 44
    assert counts["topk_sort_stages"] == 80
    assert counts["topk_cas_lanes"] == 256 * 80
    assert counts["selected_valid_indices"] == 16
    assert counts["selected_padded_indices"] == 16
    assert counts["selected_index_bytes"] == 64
    assert counts["score_workspace_bytes"] == 1 * 1 * 272 * 4
    assert counts["hbm_read_bytes"] == (
        counts["score_q_bytes"] + counts["score_k_bytes"]
        + counts["score_block_table_bytes"] + counts["score_read_bytes"]
    )
    assert counts["mte3_bytes"] == counts["score_store_bytes"] + 64


def test_score_store_counts_invisible_blocks_in_last_tile():
    params = _base_params(query=128, context=100, topk=8, topk_block=64)
    counts = dict(subject.make_op(params).metadata)["work_components"]
    # tile 0: 64 rows, blocks = ceil(min(228, 164)/128) = 2
    # tile 1: 64 rows, blocks = ceil(228/128) = 2
    assert counts["score_tiles"] == 2
    assert counts["score_store_bytes"] == (64 * 2 + 64 * 2) * 4
    # v_j = 1 for tokens 0..27 and 2 for tokens 28..127.
    assert counts["valid_visible_scores"] == 28 * 1 + 100 * 2
    assert counts["score_read_bytes"] == (28 * 1 + 100 * 2) * 4
    # The score launch writes the -inf score of a block the early tokens cannot
    # see, so its store traffic exceeds the later top-k load traffic.
    assert counts["score_store_bytes"] > counts["score_read_bytes"]
    assert counts["score_mask_lanes"] == 3 * 64 * 128


def test_score_workspace_address_map_and_roundtrip():
    params = _base_params(query=65, context=100, topk=4, topk_block=64,
                          batch=2, heads=1)
    counts = dict(subject.make_op(params).metadata)["work_components"]
    writes, reads, stride = _score_addresses(params)
    assert counts["score_stride"] == stride
    assert reads <= writes
    assert len(writes) * 4 == counts["score_store_bytes"]
    assert len(reads) * 4 == counts["score_read_bytes"]
    assert counts["score_workspace_bytes"] == 2 * 65 * stride * 4
    assert max(writes) < counts["score_workspace_bytes"]
    # Unwritten stride padding must not appear in either address set.
    assert all((address // 4) % stride < counts["max_blocks"]
               for address in writes)
    outputs = _output_addresses(params)
    assert len(outputs) * 4 == counts["selected_index_bytes"]
    assert counts["selected_index_bytes"] == 2 * 65 * 4 * 4


def test_paged_k_keys_are_reread_once_per_query_tile():
    params = _base_params(query=65, context=1024, topk=16, topk_block=64)
    counts = dict(subject.make_op(params).metadata)["work_components"]
    blocks = _ceil_div(1024 + 65, SPARSE)
    assert counts["score_tiles"] == 2
    assert blocks == counts["max_blocks"] == 9
    # Every tile loops to the same final page and reloads the full 128x128 page.
    assert counts["score_k_bytes"] == 2 * blocks * SPARSE * 128 * 2
    rng = np.random.default_rng(5)
    table = rng.permutation(blocks)
    touched = set()
    for _tile in range(2):
        for block in range(blocks):
            page = int(table[block])
            for position in range(SPARSE):
                for dim in range(128):
                    touched.add(page * SPARSE * 128 * 2 + position * 128 * 2 + dim * 2)
    assert len(touched) == blocks * SPARSE * 128
    assert 2 * len(touched) * 2 == counts["score_k_bytes"]


def test_batch_and_heads_scale_all_physical_work():
    base = _base_params(query=3, context=640, topk=2)
    scaled = _base_params(query=3, context=640, topk=2, batch=3, heads=2)
    left = dict(subject.make_op(base).metadata)["work_components"]
    right = dict(subject.make_op(scaled).metadata)["work_components"]
    for key in ("score_qk_flops", "score_q_bytes", "score_k_bytes",
                "score_block_table_bytes", "score_store_bytes",
                "score_read_bytes", "score_mask_lanes", "score_max_lanes",
                "score_addr_lanes", "topk_iterations", "topk_padded_lanes",
                "topk_sort_stages", "topk_cas_lanes", "topk_merge_lanes",
                "topk_prepare_ops", "topk_sort_ops", "topk_merge_ops",
                "topk_init_ops", "topk_extract_ops", "vector_ops",
                "valid_visible_scores", "selected_valid_indices",
                "selected_valid_bytes", "selected_index_bytes",
                "score_workspace_bytes", "hbm_read_bytes", "mte2_bytes",
                "mte3_bytes"):
        assert right[key] == 6 * left[key], key
    for key in ("score_tiles", "max_blocks", "score_stride", "block_size_t",
                "topk_first_iteration_stages", "topk_later_iteration_stages"):
        assert right[key] == left[key], key
    assert right["score_programs"] == 6 * left["score_programs"]
    assert right["topk_programs"] == 6 * left["topk_programs"]


def test_explicit_topk_tile_changes_padded_sort_work():
    common = dict(query=1, context=32768, topk=16)
    small = dict(subject.make_op(_base_params(topk_block=64, **common)).metadata)[
        "work_components"]
    medium = dict(subject.make_op(_base_params(topk_block=256, **common)).metadata)[
        "work_components"]
    large = dict(subject.make_op(_base_params(topk_block=1024, **common)).metadata)[
        "work_components"]
    assert small["topk_iterations"] == 5
    assert small["topk_padded_lanes"] == 320
    assert small["topk_sort_stages"] == 21 + 4 * 27
    assert medium["topk_iterations"] == 2
    assert medium["topk_padded_lanes"] == 512
    assert medium["topk_sort_stages"] == 36 + 44
    assert large["topk_iterations"] == 1
    assert large["topk_padded_lanes"] == 1024
    assert large["topk_sort_stages"] == 55
    assert (dict(subject.make_op(_base_params(topk_block=64, **common)).metadata)
            ["dispatch"]["topk_block"] == 64)


# ---------------------------------------------------------------------------
# Source-path guards and bounded construction.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"implementation": "vendor_topk"}, "unsupported_index_topk_implementation"),
        ({"dtype": "fp16"}, "unsupported_index_topk_dtype"),
        ({"head_dim": 64}, "unsupported_index_head_dim"),
        ({"score_block": 64}, "unsupported_sparse_block"),
        ({"block_size_q": 4}, "unsupported_block_size_q"),
        ({"topk_block": 100}, "unsupported_topk_tile"),
        ({"topk_block": 64, "topk": 64}, "index_topk_tile_must_exceed_padded_topk"),
        ({"query": 0}, "index_topk_query_must_be_positive"),
        ({"context": -1}, "index_topk_context_must_be_nonnegative"),
        ({"topk": 0}, "index_topk_topk_must_be_positive"),
        ({"batch": 0}, "index_topk_batch_and_heads_must_be_positive"),
        ({"local_blocks": -1}, "index_topk_forced_blocks_must_be_nonnegative"),
        ({"unknown_flag": True}, "index_topk_unknown_parameter"),
        ({"query": True}, "index_topk_parameter_must_be_integer:query"),
        ({"mask_local": 1}, "index_topk_parameter_must_be_boolean:mask_local"),
    ],
)
def test_unsupported_selected_paths_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        subject.make_op(_base_params(**overrides))


def test_build_rejects_inconsistent_metadata_and_other_implementation():
    op = subject.make_op(_base_params())
    meta = dict(op.metadata)
    bad_counts = dict(meta["work_components"])
    bad_counts["score_qk_flops"] += 1
    tampered = replace(op, metadata=tuple(
        (key, bad_counts if key == "work_components" else value)
        for key, value in op.metadata
    ))
    with pytest.raises(ValueError, match="index_topk_metadata_counts_inconsistent"):
        subject.build(tampered, Geometry(32, {"vector": 64}, {}))

    bad_dispatch = dict(meta["dispatch"])
    bad_dispatch["implementation"] = "vendor_topk"
    tampered = replace(op, metadata=tuple(
        (key, bad_dispatch if key == "dispatch" else value)
        for key, value in op.metadata
    ))
    with pytest.raises(ValueError, match="unsupported_index_topk_implementation"):
        subject.build(tampered, Geometry(32, {"vector": 64}, {}))

    with pytest.raises(ValueError, match="index_topk_compute_units_must_be_positive"):
        subject.build(op, Geometry(0, {"vector": 64}, {}))


def test_large_token_geometry_is_bounded_without_per_token_events(monkeypatch):
    params = _base_params(query=200000, context=1000000, topk=16,
                          topk_block=256)
    op = subject.make_op(params)
    counts = dict(op.metadata)["work_components"]
    assert counts["topk_programs"] == 200000
    assert counts["topk_iterations"] > 200000
    assert counts["topk_padded_lanes"] % 256 == 0
    plan = subject.build(op, Geometry(32, {"vector": 64}, {}))
    context = _recipe_context(op)

    calls = {"topk": 0, "score": 0}
    real_topk = subject._topk_token_stats
    real_score = subject._score_tile_stats

    def counted_topk(g, token):
        calls["topk"] += 1
        return real_topk(g, token)

    def counted_score(g, tile):
        calls["score"] += 1
        return real_score(g, tile)

    monkeypatch.setattr(subject, "_topk_token_stats", counted_topk)
    monkeypatch.setattr(subject, "_score_tile_stats", counted_score)
    program = plan.program(context)
    # Building the plan and its finite CTA streams does not inspect a token.
    assert calls == {"topk": 0, "score": 0}
    score_streams = _launch_streams(program, "score_launch")
    topk_streams = _launch_streams(program, "topk_launch")
    assert len(score_streams) == 32
    assert len(topk_streams) == 32
    assert sum(stream.count for stream in topk_streams) == 200000
    assert calls == {"topk": 0, "score": 0}
    # The planner stores a fixed-size record, not one event per token.
    serialized = json.dumps(dict(op.metadata), sort_keys=True)
    assert len(serialized) < 20000
