"""Independent math, address, tiling and resource checks for the native FN.

These tests deliberately do not call the planner's chunk builder to obtain an
expected numerical result.  The direct definition of causal convolution is
compared with a separate source-schedule interpreter that only consumes public
parameters and the public host-tiling projection.
"""
from __future__ import annotations

import json
from pathlib import Path
import tracemalloc

import numpy as np
import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operators import native_causal_conv_prefill as native
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan, RequestView

CASES = Path(__file__).parents[2] / "tests/fixtures/operator_cases/native_causal_conv_prefill.json"
SENTINEL_Y = -98765.0
SENTINEL_STATE = -12345.0


def _control_case(*, split_copy=False, units=32):
    from mindie_sim.prediction.instructions import REGISTER_RESOURCES
    rates = {"memory": 1e12, "vector": 1e12}
    rates.update(dict.fromkeys(REGISTER_RESOURCES, 1e12))
    if split_copy:
        rates.update(read=1e12, write=1e12)
    return {
        "machine": {
            "name": "synthetic-contract-control", "num_devices": 1,
            "device": {"memory_bytes": 64 * 1024**3, "compute_units": units,
                       "compute_peaks": {"bf16": 1e14, "fp16": 1e14,
                                         "fp32": 2.5e13, "fp8": 2e14,
                                         "int8": 2e14, "tf32": 5e13},
                       "rates": rates, "resource_units": {
                           "vector": 2 * units, **dict.fromkeys(REGISTER_RESOURCES, 2 * units)},
                       "service_modes": {"reg_issue_slots": {"resource": "vector", "rate": 1e12}}},
        },
        "model": {"name": "unlabeled-geometry", "layers": 1, "hidden_size": 8,
                  "num_heads": 1, "num_kv_heads": 1, "head_dim": 8,
                  "intermediate_size": 16, "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1,
                                  "output_tokens": 1}]},
        "runtime": {},
    }



def _cases():
    return json.loads(CASES.read_text())["cases"]


def _by_id(case_id):
    return next(case for case in _cases() if case["id"] == case_id)


def _single_aiv():
    return Geometry(32, {"vector": 1}, {"ub": native.DEFAULT_UB_BYTES}, {"ub": 1})


def _geometry(units=32, vector=None, ub=None):
    vector = 2 * units if vector is None else vector
    buffers = {"ub": ub} if ub is not None else {}
    return Geometry(units, {"vector": vector}, buffers, {"ub": vector} if buffers else {})


def _cast(values, dtype):
    values = np.asarray(values, dtype=np.float32)
    if dtype == "fp16":
        return values.astype(np.float16).astype(np.float32)
    bits = values.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


def _silu(values):
    with np.errstate(over="ignore"):
        return values / (np.float32(1.0) + np.exp(-values))


def _skip(cache, params):
    if cache == params.get("pad_slot_id", -1):
        return True
    null = params.get("null_block_id", 0)
    return null is not None and cache == null


def _cache_indices(params, lengths):
    raw = params.get("cache_indices")
    return tuple(range(len(lengths))) if raw is None else tuple(raw)


def _initial_state(params, state_rows, width, rng, sentinel=False):
    slots = int(params.get("state_history_slots", width - 1))
    shape = (state_rows, slots, params["dim"])
    if sentinel:
        return np.full(shape, SENTINEL_STATE, dtype=np.float32)
    values = rng.normal(0, 0.5, size=shape).astype(np.float32)
    return _cast(values, params.get("dtype", "bf16"))


def _inputs(params, state_rows, rng, sentinel_state=False):
    lengths = tuple(int(v) for v in params["lengths"])
    dim, width = int(params["dim"]), int(params["width"])
    total = sum(lengths)
    dtype = params.get("dtype", "bf16")
    x = _cast(rng.normal(0, 0.5, size=(total, dim)), dtype)
    weight = _cast(rng.normal(0, 0.5, size=(width, dim)), dtype)
    bias = (_cast(rng.normal(0, 0.2, size=(dim,)), dtype)
            if params.get("bias", False) else np.zeros(dim, dtype=np.float32))
    state = _initial_state(params, state_rows, width, rng, sentinel_state)
    return x, weight, bias, state


def _direct_reference(params, x, weight, bias, state):
    """Mathematical definition, computed without the source chunk schedule."""
    lengths = tuple(int(v) for v in params["lengths"])
    dim, width = int(params["dim"]), int(params["width"])
    dtype = params.get("dtype", "bf16")
    cache = _cache_indices(params, lengths)
    initial = params.get("has_initial_state")
    activation = params.get("activation")
    qsl = [0]
    for length in lengths:
        qsl.append(qsl[-1] + length)
    y = np.full((qsl[-1], dim), SENTINEL_Y, dtype=np.float32)
    result_state = state.copy()
    for seq, length in enumerate(lengths):
        if length == 0 or _skip(cache[seq], params):
            continue
        has_init = bool(initial[seq]) if initial is not None else False
        history = (state[cache[seq], :width - 1].astype(np.float32)
                   if has_init else np.zeros((width - 1, dim), dtype=np.float32))
        extended = np.vstack([history, x[qsl[seq]:qsl[seq] + length]])
        out = np.empty((length, dim), dtype=np.float32)
        for token in range(length):
            out[token] = bias + sum(
                weight[tap] * extended[token + tap] for tap in range(width))
        if activation in {"silu", "swish"}:
            out = _silu(out)
        y[qsl[seq]:qsl[seq] + length] = _cast(out, dtype)
        if state.shape[0] > 0:
            # Last width-1 rows of history + sequence, regardless of chunking.
            # Extra convStates slots beyond width-1 remain untouched.
            result_state[cache[seq], :width - 1] = extended[length:length + width - 1]
    return y, result_state


def _source_schedule(params, x, weight, bias, state):
    """Independent interpreter of InitRingSeqSplit / RunSeq / tail writeback."""
    op = native.make_op(params)
    host = native.host_tiling(op)
    lengths = tuple(int(v) for v in params["lengths"])
    dim, width = int(params["dim"]), int(params["width"])
    dtype = params.get("dtype", "bf16")
    cache = _cache_indices(params, lengths)
    initial = params.get("has_initial_state")
    activation = params.get("activation")
    qsl = [0]
    for length in lengths:
        qsl.append(qsl[-1] + length)
    y = np.full((qsl[-1], dim), SENTINEL_Y, dtype=np.float32)
    result_state = state.copy()
    history_count = width - 1
    token_block_size = host["token_block_size"]
    for seq, length in enumerate(lengths):
        if length == 0 or _skip(cache[seq], params):
            continue
        has_init = bool(initial[seq]) if initial is not None else False
        cursor = qsl[seq]
        end = qsl[seq + 1]
        while cursor < end:
            tile_end = min(end, (cursor // token_block_size + 1) * token_block_size)
            history = []
            for index in range(history_count):
                source_token = cursor - history_count + index
                if source_token >= qsl[seq]:
                    history.append(x[source_token])
                elif has_init:
                    state_pos = source_token - qsl[seq] + history_count
                    history.append(state[cache[seq], state_pos])
                else:
                    history.append(np.zeros(dim, dtype=np.float32))
            extended = np.vstack([np.asarray(history, dtype=np.float32),
                                  x[cursor:tile_end]])
            out = np.empty((tile_end - cursor, dim), dtype=np.float32)
            for token in range(tile_end - cursor):
                out[token] = bias + sum(
                    weight[tap] * extended[token + tap] for tap in range(width))
            if activation in {"silu", "swish"}:
                out = _silu(out)
            y[cursor:tile_end] = _cast(out, dtype)
            cursor = tile_end
        if state.shape[0] > 0:
            if length >= history_count:
                result_state[cache[seq], :history_count] = x[end - history_count:end]
            else:
                history = (state[cache[seq], :history_count].astype(np.float32)
                           if has_init else np.zeros((history_count, dim), dtype=np.float32))
                extended = np.vstack([history, x[qsl[seq]:end]])
                result_state[cache[seq], :history_count] = extended[
                    length:length + history_count]
    return y, result_state



@pytest.mark.parametrize("override,match", [
    ({"run_mode": 1}, "runMode=0"),
    ({"head_num": 1}, "head-major"),
    ({"input_mode": "batch"}, "varlen"),
    ({"num_accepted_tokens": [1]}, "numAcceptedTokens"),
])
def test_unsupported_source_branches_are_rejected_not_substituted(override, match):
    params = {
        "lengths": [2], "dim": 16, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "has_initial_state": [False], "cache_indices": [1],
        "state_history_slots": 6,
    }
    params.update(override)
    with pytest.raises(ValueError, match=match):
        native.make_op(params)


def test_host_tiling_matches_the_fixed_coordinator_oracles_and_dispatch():
    for case in _cases():
        op = native.make_op(case["params"])
        dispatch = dict(op.metadata)["dispatch"]
        assert dispatch["implementation"] == "causal_conv1d_arch35_fn"
        assert dispatch["activation"] == case["expected_dispatch"]["activation"]
        assert dispatch["bias"] is case["expected_dispatch"]["bias"]
        assert op.recipe == "causal_conv1d_arch35_fn"
        assert dict(op.metadata)["source_kernel_launches"] == 1
        assert not hasattr(native, "kernels")
        if "expected_planner" in case:
            host = native.host_tiling(op)
            selected = {key: host[key] for key in case["expected_planner"]}
            assert selected == case["expected_planner"], case["id"]


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
def test_width_activation_and_chunk_boundaries_match_direct_convolution(width, activation):
    params = {
        "lengths": [1, 2, 3, 5], "dim": 16, "width": width,
        "dtype": "bf16", "activation": activation, "bias": False,
        "metadata_placement": "cpu",
        "has_initial_state": [True, False, True, False],
        "cache_indices": [1, 2, 4, 3], "state_history_slots": 6,
        "state_rows": 6,
    }
    rng = np.random.default_rng(1000 + width * 10 + (activation == "silu"))
    x, weight, bias, state = _inputs(params, 6, rng)
    expected_y, expected_state = _direct_reference(params, x, weight, bias, state)
    actual_y, actual_state = _source_schedule(params, x, weight, bias, state)
    np.testing.assert_allclose(actual_y, expected_y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_state, expected_state, rtol=1e-6, atol=1e-6)
    # Every active token row is written once; no inactive row is initialized.
    active = [seq for seq, length in enumerate(params["lengths"])
              if length and not _skip(params["cache_indices"][seq], params)]
    active_tokens = sum(params["lengths"][seq] for seq in active)
    assert int(np.count_nonzero(actual_y[:, 0] != SENTINEL_Y)) == active_tokens


def test_float16_uses_the_same_native_branch_with_fp16_output_rounding():
    params = {
        "lengths": [3, 2], "dim": 16, "width": 3, "dtype": "fp16",
        "activation": "silu", "bias": False, "metadata_placement": "cpu",
        "has_initial_state": [True, False], "cache_indices": [1, 2],
        "state_history_slots": 6, "state_rows": 4,
    }
    rng = np.random.default_rng(4242)
    x, weight, bias, state = _inputs(params, 4, rng)
    expected_y, expected_state = _direct_reference(params, x, weight, bias, state)
    actual_y, actual_state = _source_schedule(params, x, weight, bias, state)
    np.testing.assert_allclose(actual_y, expected_y, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(actual_state, expected_state, rtol=1e-3, atol=1e-3)
    op = native.make_op(params)
    assert dict(op.metadata)["dispatch"]["dtype"] == "fp16"
    fp16 = native.source_ledger(op)
    bf16 = native.source_ledger(native.make_op(dict(params, dtype="bf16")))
    assert fp16["work"] == bf16["work"]
    assert fp16["metadata_qsl_bytes"] == bf16["metadata_qsl_bytes"]


@pytest.mark.parametrize("width", [2, 3, 4])
def test_short_sequence_state_tail_follows_source_history_slots(width):
    params = {
        "lengths": [1], "dim": 16, "width": width, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "cpu",
        "has_initial_state": [True], "cache_indices": [2],
        "state_history_slots": 6, "state_rows": 5,
    }
    rng = np.random.default_rng(200 + width)
    x, weight, bias, state = _inputs(params, 5, rng)
    actual_y, actual_state = _source_schedule(params, x, weight, bias, state)
    history_count = width - 1
    expected_state = np.vstack([state[2][1:history_count], x[0:1]])
    np.testing.assert_allclose(actual_state[2, :history_count], expected_state,
                               rtol=1e-6, atol=1e-6)
    # Untouched cache rows and the extra convStates slots remain unchanged.
    for row in (0, 1, 3, 4):
        np.testing.assert_array_equal(actual_state[row], state[row])
    np.testing.assert_array_equal(actual_state[2, history_count:],
                                  state[2, history_count:])



def test_bias_generic_branch_matches_direct_convolution_and_writes_state():
    params = {
        "lengths": [2, 5], "dim": 32, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": True, "metadata_placement": "cpu",
        "has_initial_state": [True, False], "cache_indices": [1, 3],
        "state_history_slots": 6, "state_rows": 5,
    }
    rng = np.random.default_rng(5150)
    x, weight, bias, state = _inputs(params, 5, rng)
    expected_y, expected_state = _direct_reference(params, x, weight, bias, state)
    actual_y, actual_state = _source_schedule(params, x, weight, bias, state)
    np.testing.assert_allclose(actual_y, expected_y, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(actual_state, expected_state, rtol=1e-6, atol=1e-6)
    op = native.make_op(params)
    assert dict(op.metadata)["dispatch"]["bias"] is True
    # Unused cache rows and extra stateLen slots stay untouched.
    np.testing.assert_array_equal(actual_state[0], state[0])
    np.testing.assert_array_equal(actual_state[1, 3:], state[1, 3:])


def test_cache_index_permutation_remaps_state_ownership_without_changing_output():
    params = {
        "lengths": [3, 2], "dim": 16, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "cpu",
        "has_initial_state": [True, True], "cache_indices": [1, 3],
        "state_history_slots": 6, "state_rows": 5,
    }
    rng = np.random.default_rng(321)
    x, weight, bias, state = _inputs(params, 5, rng)
    y_a, state_a = _source_schedule(params, x, weight, bias, state)
    swapped = dict(params, cache_indices=[3, 1])
    permuted = state.copy()
    permuted[1], permuted[3] = state[3].copy(), state[1].copy()
    y_b, state_b = _source_schedule(swapped, x, weight, bias, permuted)
    np.testing.assert_allclose(y_a, y_b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_a[1], state_b[3], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(state_a[3], state_b[1], rtol=1e-6, atol=1e-6)
    for row in (0, 2, 4):
        np.testing.assert_array_equal(state_a[row], state[row])


def test_null_pad_and_empty_sequences_leave_storage_unspecified_not_zero():
    params = {
        "lengths": [1, 2, 0, 5, 3], "dim": 16, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "has_initial_state": [True, True, False, True, False],
        "cache_indices": [1, 0, 3, -1, 4], "state_history_slots": 6,
        "state_rows": 8, "null_block_id": 0, "pad_slot_id": -1,
    }
    rng = np.random.default_rng(777)
    x, weight, bias, state = _inputs(params, 8, rng, sentinel_state=True)
    y, updated = _source_schedule(params, x, weight, bias, state)
    qsl = [0]
    for length in params["lengths"]:
        qsl.append(qsl[-1] + length)
    # seq1 (cache 0) and seq3 (cache -1) are skipped; seq2 is empty.
    for seq in (1, 3):
        assert np.all(y[qsl[seq]:qsl[seq + 1]] == SENTINEL_Y)
    assert np.all(y[0:1] != SENTINEL_Y)
    assert np.all(y[6:8] == SENTINEL_Y)
    assert np.all(y[8:11] != SENTINEL_Y)
    for row in (0, 2, 3, 5, 6, 7):
        np.testing.assert_array_equal(updated[row], state[row])
    assert not np.all(updated[1] == SENTINEL_STATE)
    assert not np.all(updated[4] == SENTINEL_STATE)
    ledger = native.source_ledger(native.make_op(params))
    assert ledger["output_write_bytes"] == 4 * 16 * 2
    assert ledger["state_write_bytes"] == 2 * 3 * 16 * 2
    assert ledger["prefetch_payload_read_bytes"] == 3 * 16 * 2


def test_no_bias_rolling_branch_has_source_vector_counts_and_device_sync():
    params = {
        "lengths": [8], "dim": 16, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "has_initial_state": [False], "cache_indices": [1],
        "state_history_slots": 6, "state_rows": 3,
    }
    ledger = native.source_ledger(native.make_op(params), _single_aiv())
    # Setup remains analytical. Restore and token VF calls consume typed
    # issue/ports/lanes, not an extra generic vector charge.
    assert ledger["work"]["vector"] == 80 + 48
    assert ledger["work"]["reg_issue_slots"] == 25 + 8 * 30
    assert ledger["work"]["reg_ub_accesses"] == 9 + 8 * 12
    # Device all-false keeps hasInitialState=1: workspace and SyncAll remain.
    assert ledger["host_tiling"]["has_initial_state"] == 1
    assert ledger["host_tiling"]["has_init_state_workspace"] == 1
    assert ledger["block_dim"] == 1
    assert ledger["sync_workspace_bytes"] == ledger["block_dim"] * 32
    assert ledger["snapshot_workspace_bytes"] == 1 * 3 * 16 * 2
    assert ledger["prefetch_payload_read_bytes"] == 0
    activation_none = dict(params, activation=None)
    less = native.source_ledger(native.make_op(activation_none), _single_aiv())
    assert ledger["work"]["vector"] == less["work"]["vector"]
    assert ledger["work"]["reg_issue_slots"] - less["work"]["reg_issue_slots"] == 4 * 8
    assert ledger["work"]["reg_special_lanes"] == 2 * 8 * 16


def test_bias_selects_generic_runseq_without_rolling_restore_or_state_reload():
    base = {
        "lengths": [8], "dim": 16, "width": 4, "dtype": "bf16",
        "activation": "silu", "metadata_placement": "device",
        "has_initial_state": [False], "cache_indices": [1],
        "state_history_slots": 6, "state_rows": 3,
    }
    rolling = native.source_ledger(native.make_op(dict(base, bias=False)), _single_aiv())
    generic = native.source_ledger(native.make_op(dict(base, bias=True)), _single_aiv())
    # Generic bias path: bias add + 4 casts + 4 MulAddDst + output cast = 18
    # element operations per token, and no rolling restore local partials.
    assert generic["work"]["vector"] == 80 + 48 + 8 * 18 * 16
    assert rolling["work"]["vector"] == 80 + 48
    assert rolling["work"]["reg_issue_slots"] == 25 + 8 * 30
    # Generic bias adds one weight row (32 B) and skips the rolling state
    # reload (3*16*2 = 96 B): rolling - generic = 64 B.
    assert rolling["work"]["mte2"] - generic["work"]["mte2"] == 3 * 16 * 2 - 16 * 2


@pytest.mark.parametrize("width,rolling_slots,restore_slots", [
    (2, 18, 7), (3, 24, 15), (4, 30, 25),
])
def test_width_changes_rolling_register_work(width, rolling_slots, restore_slots):
    params = {
        "lengths": [8], "dim": 16, "width": width, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "has_initial_state": [False], "cache_indices": [1],
        "state_history_slots": 6, "state_rows": 3,
    }
    ledger = native.source_ledger(native.make_op(params), _single_aiv())
    history_zero = (4 - width + width - 1) * 16
    assert ledger["work"]["vector"] == 80 + history_zero
    assert ledger["work"]["reg_issue_slots"] == restore_slots + 8 * rolling_slots
    assert ledger["work"]["reg_ub_stores"] == width - 1 + 8 * width



def test_sequence_boundary_history_reload_and_snapshot_payload_ledger():
    base = {
        "lengths": [8], "dim": 16, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "cache_indices": [1], "state_history_slots": 6, "state_rows": 3,
    }
    false_init = native.source_ledger(native.make_op(
        dict(base, has_initial_state=[False])))
    # TokenBlockSize=1 at cuSeqlen=8/core=64.  Chunk c has min(c, 3) preceding
    # x tokens; the other history slots are zero.  Six zero-fill vectors total
    # (3 for block 0, 2 for block 1, 1 for block 2).
    weight = 8 * 4 * 16 * 2
    history_and_first = (32 + 64 + 96 + 5 * 128)
    state_read = 3 * 16 * 2
    state_write = 3 * 16 * 2
    output = 8 * 16 * 2
    chunk_metadata = 8 * 17
    prefetch_his = 1
    sync = 8 * 32
    assert false_init["work"]["mte2"] == (weight + history_and_first
                                          + state_read + chunk_metadata + prefetch_his)
    assert false_init["work"]["mte3"] == output + state_write
    assert false_init["work"]["hbm"] == (weight + history_and_first + state_read
                                          + chunk_metadata + prefetch_his
                                          + output + state_write + sync)
    assert false_init["work"]["vector"] == 8 * 80 + 6 * 16
    assert false_init["work"]["reg_issue_slots"] == 8 * 25 + 8 * 30
    assert false_init["prefetch_payload_read_bytes"] == 0
    assert false_init["snapshot_workspace_bytes"] == 3 * 16 * 2

    true_init = native.source_ledger(native.make_op(
        dict(base, has_initial_state=[True])))
    assert true_init["prefetch_payload_read_bytes"] == 3 * 16 * 2
    assert true_init["prefetch_payload_write_bytes"] == 3 * 16 * 2
    assert true_init["snapshot_workspace_bytes"] == 3 * 16 * 2
    # Every pre-sequence history slot in blocks 0..2 now reads the snapshot:
    # 3+2+1 = 6 slots, i.e. +192 read bytes over zero fill.  The prefetch load
    # adds payload 96 plus the new cache-index read 4; the store adds 96.
    assert true_init["work"]["mte2"] - false_init["work"]["mte2"] == 192 + 96 + 4
    assert true_init["work"]["mte3"] - false_init["work"]["mte3"] == 96
    assert true_init["work"]["vector"] == false_init["work"]["vector"] - 6 * 16
    assert true_init["work"]["hbm"] - false_init["work"]["hbm"] == 192 + 96 + 4 + 96


def test_device_all_false_and_cpu_all_false_have_distinct_workspace_and_metadata():
    device = native.source_ledger(native.make_op(_by_id("all_false_device")["params"]))
    cpu = native.source_ledger(native.make_op(_by_id("all_false_cpu")["params"]))
    assert device["host_tiling"]["has_initial_state"] == 1
    assert device["host_tiling"]["has_initial_state"] == 1
    assert device["sync_workspace_bytes"] == device["block_dim"] * 32
    assert device["snapshot_workspace_bytes"] == 3 * 3 * 16 * 2
    assert device["prefetch_payload_read_bytes"] == 0
    assert cpu["host_tiling"]["has_initial_state"] == 0
    assert cpu["host_tiling"]["has_init_state_workspace"] == 0
    assert cpu["sync_workspace_bytes"] == 0
    assert cpu["snapshot_workspace_bytes"] == 0
    assert cpu["prefetch_payload_read_bytes"] == 0
    # Device int32 metadata is 4 bytes; CPU int64 is 8 bytes.  The CPU qsl
    # branch uses explicit token ranges (no binary search) and must not be
    # charged as eight-byte device loads.
    assert device["metadata_qsl_bytes"] == 176
    assert cpu["metadata_qsl_bytes"] == 176
    assert device["metadata_cache_bytes"] == 44
    assert cpu["metadata_cache_bytes"] == 88
    assert device["metadata_has_initial_state_bytes"] == 14
    assert cpu["metadata_has_initial_state_bytes"] == 0


def test_metadata_loads_follow_token_tile_and_prefetch_scope():
    ledger = native.source_ledger(native.make_op(_by_id("token_threshold")["params"]))
    # 64 blocks: one 1-read binary search (batch=1) + two qsl reads per seq.
    assert ledger["metadata_qsl_bytes"] == 64 * (1 + 2) * 4
    assert ledger["metadata_cache_bytes"] == 64 * 4
    # 64 chunk-level bool reads plus one prefetch-owner bool read.
    assert ledger["metadata_has_initial_state_bytes"] == 64 + 1
    assert ledger["has_initial_state_flag"] == 1
    assert ledger["prefetch_payload_read_bytes"] == 0
    assert ledger["sync_workspace_bytes"] == 64 * 32


def test_aiv_ownership_is_the_declared_vector_unit_count_not_aic_count():
    op = native.make_op(_by_id("token_threshold")["params"])
    small_aic = native.host_tiling(op, _geometry(units=32, vector=64))
    large_aic = native.host_tiling(op, _geometry(units=64, vector=64))
    assert small_aic["block_dim"] == large_aic["block_dim"] == 64
    fewer_aiv = native.host_tiling(op, _geometry(units=32, vector=32))
    assert fewer_aiv["block_dim"] == 32
    assert fewer_aiv["core_count"] == 32
    assert native.host_tiling(op)["core_count"] == 64


def test_explicit_ub_geometry_changes_cutbsd_base_dim_and_small_ub_rejects_build():
    op = native.make_op(_by_id("nondivisor_dim_cosplit")["params"])
    default = native.host_tiling(op)
    explicit = native.host_tiling(op, _geometry(units=32, vector=64, ub=253952))
    smaller = native.host_tiling(op, _geometry(units=32, vector=64, ub=100000))
    assert default["base_dim"] == explicit["base_dim"] == 2560
    assert smaller["valid"] and smaller["base_dim"] < default["base_dim"]
    with pytest.raises(ValueError, match="UB capacity"):
        native.build(op, _geometry(units=32, vector=64, ub=100000))


def test_actual_execution_conserves_declared_work_in_both_memory_profiles():
    op = native.make_op(_by_id("all_false_device")["params"])
    ledger = native.source_ledger(op)
    for split in (False, True):
        result = verify_operator(_control_case(split_copy=split, units=32), op, native.build)
        assert result["status"] == "passed"
        assert result["buffers_released"] and result["finite_completion"]
        assert result["work_mismatches"] == {}
        row = result["result"]
        assert row["operator"]["vector_ops"] == ledger["work"]["vector"]
        assert row["operator"]["hbm_bytes"] == ledger["work"]["hbm"]
        # This synthetic mode uses the same numeric base rate for issue.
        assert row["serviced"]["vector@r0:s0"] == (
            row["operator"]["vector_ops"] + ledger["work"]["reg_issue_slots"])
        assert row["serviced"]["memory@r0:s0"] == row["operator"]["hbm_bytes"]
        if split:
            assert row["serviced"]["mte2@r0:s0"] == ledger["work"]["mte2"]
            assert row["serviced"]["mte3@r0:s0"] == ledger["work"]["mte3"]




def test_host_grid_that_cannot_cover_every_dim_tile_is_rejected():
    # baseDim caps at 4096, so dim > 4096*64 needs 74 channel tiles but the
    # source launch width clamps blockDim to 64.  Refuse rather than silently
    # leaving output channels unwritten.
    params = {
        "lengths": [1], "dim": 300000, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "has_initial_state": [False], "cache_indices": [1],
        "state_history_slots": 6,
    }
    op = native.make_op(params)
    host = native.host_tiling(op)
    assert host["base_dim"] == 4096
    assert host["base_dim_count"] > host["core_count"]
    assert host["block_dim"] < host["grid_size"]
    with pytest.raises(ValueError, match="fewer blocks"):
        native.build(op, _geometry(units=32, vector=64))


def test_declared_ub_pool_is_reserved_per_aiv_and_released():
    op = native.make_op(_by_id("one_cta")["params"])
    hardware = _control_case(units=32)
    hardware["machine"]["device"]["buffers"] = {"ub": native.DEFAULT_UB_BYTES}
    hardware["machine"]["device"]["buffer_units"] = {"ub": 64}
    result = verify_operator(hardware, op, native.build)
    assert result["status"] == "passed"
    assert result["buffers_released"]
    row = result["result"]
    assert not any(row["remaining_buffers"].values())
    assert list(row["peak_buffers_per_unit"].values()) == [
        native.UB_RESIDENT_BYTES]


def test_missing_conv_states_still_skip_null_pad_but_write_no_state():
    params = {
        "lengths": [2, 3], "dim": 16, "width": 3, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "has_initial_state": [True, False], "cache_indices": [0, 1],
        "state_history_slots": 0, "state_rows": 0,
    }
    op = native.make_op(params)
    ledger = native.source_ledger(op)
    assert ledger["has_conv_states"] is False
    assert ledger["state_write_bytes"] == 0
    assert ledger["prefetch_payload_read_bytes"] == 0
    assert ledger["host_tiling"]["has_init_state_workspace"] == 0
    assert ledger["output_write_bytes"] == 3 * 16 * 2


def test_explicit_cpu_token_ranges_disable_device_binary_search_reads():
    cpu = native.source_ledger(native.make_op(_by_id("all_false_cpu")["params"]))
    device = native.source_ledger(native.make_op(_by_id("all_false_device")["params"]))
    assert cpu["host_tiling"]["has_explicit_token_seq_ranges"] == 1
    assert device["host_tiling"]["has_explicit_token_seq_ranges"] == 0
    # The CPU branch still performs sequence-window qsl reads; it merely avoids
    # the per-block binary search.  int64 bytes are explicit, not inferred.
    assert cpu["metadata_qsl_bytes"] == 11 * 2 * 8
    assert device["metadata_qsl_bytes"] == 11 * (2 + 2) * 4


def test_16k_full_case_completes_with_bounded_live_memory():
    op = native.make_op(_by_id("nondivisor_dim_cosplit")["params"])
    tracemalloc.start()
    try:
        result = verify_operator(_control_case(units=32), op, native.build)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result["status"] == "passed"
    assert result["work_mismatches"] == {}
    assert peak < 512 * 1024 * 1024


def test_large_sequence_planner_is_symbolic_not_per_token():
    params = {
        "lengths": [1048576], "dim": 10240, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "has_initial_state": [False], "cache_indices": [1],
        "state_history_slots": 6,
    }
    tracemalloc.start()
    try:
        op = native.make_op(params)
        host = native.host_tiling(op)
        ledger = native.source_ledger(op)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert host["execution_plan"] == "CUTBSD"
    assert host["base_dim"] == 2560 and host["base_dim_count"] == 4
    assert host["token_block_size"] == 65536
    assert host["token_block_count"] == 16
    assert ledger["block_dim"] == 64
    assert ledger["chunk_count"] == 64
    # One symbolic pair segment per two tokens per AIV, never materialized.
    assert ledger["segment_count"] == 64 * (65536 // 2)
    assert all(np.isfinite(value) for value in ledger["work"].values())
    assert peak < 512 * 1024 * 1024


def _trace_hardware(*, units=1, read=1e12, write=1e12, vector=1e12):
    hardware = _control_case(units=units, split_copy=True)
    device = hardware["machine"]["device"]
    device["resource_units"] = {"vector": units, "mte2": units, "mte3": units}
    device["rates"].update(read=read, write=write, vector=vector, memory=1e20)
    return hardware


def _execute(params, hardware, *, trace_limit=20000):
    op = native.make_op(params)
    case = parse_case(hardware)
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(native.build)})
    execution = Execution(case, CostModel(case), recipes=recipes,
                          trace_limit=trace_limit)
    iteration = IterationWork(
        (op,), IterationPlan((RequestView("r", sum(params["lengths"]), 0,
                                          True, "prefill"),)),
        1, 1, 1, None, "prefill", "review")
    step = execution.add(iteration, 0, 0)
    execution.engine.advance(until=(step.completion,))
    return execution.engine.trace, execution.engine.summary()


def _role_bytes(row, role):
    return sum(value for key, value in row["demand"].items()
               if key.startswith(role + "@"))


def _tile_stages(rows):
    stages = {}
    for row in rows:
        marker = ":pairs:program:tile"
        if marker not in row["task"]:
            continue
        index, label = row["task"].split(marker, 1)[1].split(":", 1)
        stages[int(index), label] = row
    return stages


@pytest.mark.parametrize("width", [2, 3, 4])
def test_rolling_state_writeback_loads_retained_input_before_state_store(width):
    """WriteBackFnState: MTE2 reload -> MTE2_MTE3 wait -> MTE3 store."""
    params = {
        "lengths": [width], "dim": 16, "width": width, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "cpu",
        "has_initial_state": [False], "cache_indices": [1],
        "state_history_slots": 6,
    }
    rows, _ = _execute(params, _trace_hardware(
        units=1, read=1e6, write=1e6, vector=1e20))
    loads = [row for row in rows if row["task"].endswith(":state_load")]
    stores = [row for row in rows if row["task"].endswith(":state_store")]
    assert len(loads) == len(stores) == 1
    # A sequence of length width reloads its last width-1 x rows:
    # (width-1)*16*2 bytes, then stores the same bytes.  No stage may carry
    # both directions at once.
    state_bytes = (width - 1) * 16 * 2
    assert _role_bytes(loads[0], "mte2") == state_bytes
    assert _role_bytes(loads[0], "mte3") == 0
    assert _role_bytes(stores[0], "mte3") == state_bytes
    assert _role_bytes(stores[0], "mte2") == 0
    # 1 MB/s in each direction: the store cannot be simultaneous with the
    # read it depends on.  For width 4 that is >= 0.192 ms for 96 B + 96 B.
    assert stores[0]["start_ms"] >= loads[0]["end_ms"] - 1e-12
    assert (stores[0]["end_ms"] - loads[0]["start_ms"]
            == pytest.approx(2 * state_bytes / 1e6 * 1000))


def test_generic_bias_state_tail_writes_ring_without_retained_x_reload():
    params = {
        "lengths": [3], "dim": 16, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": True, "metadata_placement": "cpu",
        "has_initial_state": [False], "cache_indices": [1],
        "state_history_slots": 6,
    }
    rows, _ = _execute(params, _trace_hardware(
        units=1, read=1e6, write=1e6, vector=1e20))
    loads = [row for row in rows if row["task"].endswith(":state_load")]
    stores = [row for row in rows if row["task"].endswith(":state_store")]
    # WriteBackState copies the last width-1 ring rows; there is no MTE2 x
    # reload in the generic branch.
    assert not loads
    assert len(stores) == 1
    assert _role_bytes(stores[0], "mte3") == 3 * 16 * 2
    assert _role_bytes(stores[0], "mte2") == 0
    output = [row for (index, label), row in _tile_stages(rows).items()
              if label == "store" and index == 2]
    assert stores[0]["start_ms"] >= output[0]["end_ms"] - 1e-12


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("width", [2, 3, 4])
def test_output_slot_reuse_waits_for_same_parity_store_and_keeps_overlap(
        width, bias):
    """Rolling and generic RunSeq both double-buffer output tokens t&1."""
    params = {
        "lengths": [8], "dim": 16, "width": width, "dtype": "bf16",
        "activation": "silu", "bias": bias, "metadata_placement": "cpu",
        "has_initial_state": [False], "cache_indices": [1],
        "state_history_slots": 6,
    }
    # 32 B store => 1 ms; input loads are effectively free. The old output
    # slot gates rolling compute, but only the final cast in generic RunSeq.
    rows, _ = _execute(params, _trace_hardware(
        units=1, read=1e20, write=32000, vector=232000))
    stages = _tile_stages(rows)
    for token in range(2, 8):
        output = stages[token, "cast" if bias else "compute"]
        old_store = stages[token - 2, "store"]
        assert output["start_ms"] >= old_store["end_ms"] - 1e-12, token
    # Stores form the MTE3 order; the next arithmetic follows the previous
    # vector stage, including the separate generic output cast.
    for token in range(1, 8):
        store = stages[token, "store"]
        prev_store = stages[token - 1, "store"]
        assert store["start_ms"] >= prev_store["end_ms"] - 1e-12
        compute = stages[token, "compute"]
        previous_vector = stages[token - 1, "cast" if bias else "compute"]
        assert compute["start_ms"] >= previous_vector["end_ms"] - 1e-12
    # Token 2 arithmetic can overlap store1. In generic RunSeq its final cast
    # may occur later when arithmetic is slower than that store.
    compute_two = stages[2, "compute"]
    store_one = stages[1, "store"]
    assert compute_two["start_ms"] < store_one["end_ms"] - 1e-12
    # The rolling pair copy is one MTE2 transaction for two tokens, and the
    # generic branch copies one token at a time.
    pair_loads = [row for (index, label), row in stages.items()
                  if label == "load"]
    if not bias:
        assert _role_bytes(stages[1, "load"], "mte2") == 2 * 16 * 2
        assert _role_bytes(stages[3, "load"], "mte2") == 2 * 16 * 2
        assert _role_bytes(stages[7, "load"], "mte2") == 16 * 2
        assert len(pair_loads) == 4
    else:
        assert all(_role_bytes(row, "mte2") == 16 * 2
                   for row in pair_loads)
        assert len(pair_loads) == 7


def test_snapshot_prefetch_read_precedes_workspace_store_and_sync():
    params = {
        "lengths": [3], "dim": 16, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "has_initial_state": [True], "cache_indices": [2],
        "state_history_slots": 6,
    }
    rows, _ = _execute(params, _trace_hardware(
        units=1, read=1e6, write=1e6, vector=1e20))
    prefetch = [row for row in rows if ":prefetch0:" in row["task"]]
    loads = [row for row in prefetch if ":read" in row["task"]]
    stores = [row for row in prefetch if ":write" in row["task"]]
    syncs = [row for row in rows if row["task"].endswith(":init_sync")]
    assert len(loads) == len(stores) == 3 and len(syncs) == 1
    # All three history slices share tmpBuf.  Its next input cannot overwrite
    # the prior output while MTE3 still consumes it (source fn_tasks:221-231).
    for index, (load, store) in enumerate(zip(loads, stores)):
        assert _role_bytes(load, "mte2") == 16 * 2
        assert _role_bytes(store, "mte3") == 16 * 2
        assert store["start_ms"] >= load["end_ms"] - 1e-12
        if index:
            assert load["start_ms"] >= stores[index - 1]["end_ms"] - 1e-12
    assert sum(_role_bytes(row, "mte2") for row in prefetch) == 96 + 1 + 4
    assert syncs[0]["start_ms"] >= stores[-1]["end_ms"] - 1e-12
    assert _role_bytes(syncs[0], "memory") == 32


@pytest.mark.parametrize("placement", ["cpu", "device"])
def test_absent_conv_states_forces_has_initial_state_zero(placement):
    """Validation clears hasInitialState and ResolveSeqHasInit returns false.

    A valid one-token sequence with a non-null cache index and an explicit true
    flag must do exactly the same physical work as the false flag because
    convStates is absent (state_history_slots=0, state_rows=0).
    """
    base = {
        "lengths": [1], "dim": 16, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": False,
        "metadata_placement": placement, "cache_indices": [1],
        "state_history_slots": 0, "state_rows": 0,
    }
    ledgers = []
    executions = []
    for flag in (False, True):
        op = native.make_op(dict(base, has_initial_state=[flag]))
        ledger = native.source_ledger(
            op, Geometry(1, {"vector": 1}, {}, {}))
        rows, summary = _execute(dict(base, has_initial_state=[flag]),
                                 _trace_hardware(units=1, read=1e6,
                                                 write=1e6, vector=1e6))
        ledgers.append(ledger)
        executions.append(({row["task"]: dict(row["demand"])
                            for row in rows if row["demand"]}, summary))
    false_ledger, true_ledger = ledgers
    assert true_ledger["work"] == false_ledger["work"]
    assert true_ledger["host_tiling"] == false_ledger["host_tiling"]
    assert true_ledger["host_tiling"]["has_initial_state"] == 0
    assert true_ledger["host_tiling"]["has_init_state_workspace"] == 0
    assert true_ledger["state_write_bytes"] == 0
    assert true_ledger["prefetch_payload_read_bytes"] == 0
    assert true_ledger["metadata_has_initial_state_bytes"] == 0
    # Equal executed work, not merely equal planner fields.
    assert executions[1][0] == executions[0][0]
    assert executions[1][1]["serviced"] == executions[0][1]["serviced"]
    assert executions[1][1]["time_ms"] == executions[0][1]["time_ms"]

    # The same normalization holds when absence is expressed with only the
    # zero-row descriptor (no state_history_slots field).
    implicit = {key: value for key, value in base.items()
                if key != "state_history_slots"}
    implicit_ledger = native.source_ledger(
        native.make_op(dict(implicit, has_initial_state=[True])),
        Geometry(1, {"vector": 1}, {}, {}))
    assert implicit_ledger["work"] == false_ledger["work"]
    assert implicit_ledger["host_tiling"]["has_initial_state"] == 0


def test_width_five_is_rejected_by_the_selected_source_validation():
    params = {
        "lengths": [2], "dim": 16, "width": 5, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "has_initial_state": [False], "cache_indices": [1],
        "state_history_slots": 6,
    }
    # causal_conv1d_tiling_validation.h rejects width outside [2,4]; there is
    # no native width-5 branch to model or substitute.
    with pytest.raises(ValueError, match=r"width must be in \[2, 4\]"):
        native.make_op(params)


def test_device_all_false_keeps_sync_but_skips_snapshot_payload_copies():
    device_params = _by_id("all_false_device")["params"]
    rows, _ = _execute(device_params, _trace_hardware(
        units=64, read=1e6, write=1e6, vector=1e20))
    prefetch = [row for row in rows if ":prefetch" in row["task"]]
    flags = [row for row in prefetch if row["task"].endswith(":flag")]
    syncs = [row for row in rows if row["task"].endswith(":init_sync")]
    # Tiling still launches one prefetch pass per block and the global sync;
    # only the payload copy is skipped because every actual flag is false.
    assert len(flags) == 3 and len(syncs) == 1
    assert not [row for row in prefetch if ":read" in row["task"] or ":write" in row["task"]]
    assert sum(_role_bytes(row, "mte2") for row in prefetch) == 3
    assert _role_bytes(syncs[0], "memory") == 11 * 32

    cpu_params = _by_id("all_false_cpu")["params"]
    cpu_rows, _ = _execute(cpu_params, _trace_hardware(
        units=64, read=1e6, write=1e6, vector=1e20))
    assert not [row for row in cpu_rows if ":prefetch" in row["task"]]
    assert not [row for row in cpu_rows
                if row["task"].endswith(":init_sync")]


def test_short_sequence_state_reload_uses_snapshot_rows_before_store():
    base = {
        "lengths": [1], "dim": 16, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "device",
        "cache_indices": [2], "state_history_slots": 6, "state_rows": 4,
    }
    true_rows, _ = _execute(dict(base, has_initial_state=[True]),
                            _trace_hardware(units=1, read=1e6, write=1e6,
                                            vector=1e6))
    false_rows, _ = _execute(dict(base, has_initial_state=[False]),
                             _trace_hardware(units=1, read=1e6, write=1e6,
                                             vector=1e6))
    def state_dirs(rows):
        load = next(row for row in rows
                    if row["task"].endswith(":state_load"))
        store = next(row for row in rows
                     if row["task"].endswith(":state_store"))
        return load, store
    true_load, true_store = state_dirs(true_rows)
    false_load, false_store = state_dirs(false_rows)
    # seqLen=1 < width-1=3: the true-state path reads two snapshot rows and
    # the one retained x row (3*32 B); the false path reads only that x row
    # and zero-fills the other two history rows.
    assert _role_bytes(true_load, "mte2") == 3 * 16 * 2
    assert _role_bytes(true_load, "vector") == 0
    assert _role_bytes(false_load, "mte2") == 16 * 2
    assert _role_bytes(false_load, "vector") == 2 * 16
    for load, store in ((true_load, true_store), (false_load, false_store)):
        assert _role_bytes(store, "mte3") == 3 * 16 * 2
        assert store["start_ms"] >= load["end_ms"] - 1e-12
