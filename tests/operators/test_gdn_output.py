"""Independent work, physical-address and ownership oracles for arch35 GDN O.

The source is the preserved c3d00a196905 image, chunk_fwd_o/arch35:
cube.h:103-180,209-304,308-511; vector.h:53-236,283-418;
common.h:232-264. Matrix and byte oracles do not call planner work helpers.
These checks establish mechanism contracts, not hardware timing accuracy.
"""
from collections import Counter

import numpy as np
import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operators import gdn_output
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator


BT, K, V = 64, 128, 128


def params(lengths=(64,), heads=12, key_heads=4, **overrides):
    return {"lengths": lengths, "heads": heads, "key_heads": key_heads,
            "dtype": "bf16", "gate_dtype": "fp32", "state_v_first": True,
            "use_exp2": True, "output_token_first": True, **overrides}


def hardware(*, units=4, buffers=None, split_copy=False):
    rates = {"memory": 1e12, "vector": 1e12, "mte1": 1e12, "fixpipe": 1e12}
    if split_copy:
        rates.update(read=1e12, write=1e12)
    return {
        "machine": {"name": "synthetic-contract", "num_devices": 1,
                    "device": {"memory_bytes": 1024**3, "compute_units": units,
                               "compute_peaks": {"bf16": 1e14, "fp32": 2.5e13},
                               "rates": rates, "resource_units": {"vector": 2 * units},
                               "buffers": buffers or {},
                               "buffer_units": {pool: 2 * units if pool == "ub" else units
                                                for pool in (buffers or {})}}},
        "model": {"name": "geometry", "layers": 1, "hidden_size": 8, "num_heads": 1,
                  "num_kv_heads": 1, "head_dim": 8, "intermediate_size": 16,
                  "vocab_size": 16},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
        "runtime": {},
    }


def chunks(lengths):
    start = 0
    for length in lengths:
        for row in range(0, length, BT):
            yield start + row, min(BT, length - row)
        start += length


def transfer_oracle(value):
    """Enumerate source input/output intervals, including within-group reuse."""
    heads, keys = value["heads"], value["key_heads"]
    ratio = heads // keys
    group_size = 3 if ratio == 3 else 4
    total_tokens = sum(value["lengths"])
    gate_bytes = 4 if value["gate_dtype"] == "fp32" else 2
    traffic = Counter()
    output_addresses = []
    for token_start, rows in chunks(value["lengths"]):
        for first in range(0, heads, group_size):
            visited = set()
            for head in range(first, min(first + group_size, heads)):
                key = head // ratio
                if key not in visited:
                    visited.add(key)
                    qk_begin = (key * total_tokens + token_start) * K
                    qk_interval = range(qk_begin, qk_begin + rows * K)
                    traffic["q"] += 2 * len(qk_interval)
                    traffic["k"] += 2 * len(qk_interval)
                traffic["g"] += gate_bytes * rows
                traffic["h"] += K * V * 2
                traffic["v"] += rows * V * 2
                traffic["direct_a_prime"] += BT * BT * 2
                # QK and QH publish full FP32 tiles, while AV publishes only
                # mActual output rows. All three destinations are owner UB.
                for tile_rows, tile_columns in ((BT, BT), (BT, V), (rows, V)):
                    traffic["fixpipe"] += tile_rows * tile_columns * 4
                for row in range(rows):
                    for channel in range(V):
                        output_addresses.append(
                            ((token_start + row) * heads + head) * V + channel)
    assert sorted(output_addresses) == list(range(total_tokens * heads * V))
    traffic["o"] = 2 * len(output_addresses)
    read = sum(traffic[key] for key in ("q", "k", "g", "h", "v"))
    return {"hbm": read + traffic["o"], "mte2": read,
            "mte3": traffic["o"] + traffic["direct_a_prime"],
            "fixpipe": traffic["fixpipe"]}, traffic


@pytest.mark.parametrize("value", [
    params(lengths=(1,), heads=3, key_heads=1),
    params(lengths=(17, 65), heads=8, key_heads=4),
    params(lengths=(63, 1, 65), heads=5, key_heads=5),
    params(lengths=(64,), heads=12, key_heads=3),
    params(lengths=(80,), gate_dtype="bf16", use_exp2=False, state_v_first=False),
])
def test_gm_tail_masks_group_reuse_and_direct_a_prime_transfer(value):
    op = gdn_output.make_op(value)
    plan = gdn_output.build(op, Geometry(4, {"vector": 8}))
    expected, intervals = transfer_oracle(value)
    for resource, amount in expected.items():
        assert plan.work.resources[resource] == amount
    assert plan.work.resources["mte2"] + plan.work.resources["mte3"] \
        - plan.work.resources["hbm"] == intervals["direct_a_prime"]
    for split in (False, True):
        report = verify_operator(hardware(split_copy=split), op, gdn_output.build)
        assert report["status"] == "passed", report


@pytest.mark.parametrize("rows", (1, 15, 16, 17, 32, 33, 48, 63, 64))
def test_av_requested_tail_is_distinct_from_physical_bf16_fractal_work(rows):
    # Catlass forwards actual m/k, but the official 3510 MMAD theory
    # formula rounds them to BF16 16x16x16 fractals for physical service.
    op = gdn_output.make_op(params(lengths=(rows,), heads=3, key_heads=1))
    work = gdn_output.build(op, Geometry(4, {"vector": 8})).work
    qk_outputs = BT * BT
    qh_outputs = BT * V
    av_outputs = rows * V
    requested = 3 * (qk_outputs * 2 * K + qh_outputs * 2 * K + av_outputs * 2 * rows)
    physical_av = sum(
        2 * 16**3 for _m in range(0, rows, 16)
        for _n in range(0, V, 16) for _k in range(0, rows, 16))
    physical = 3 * (qk_outputs * 2 * K + qh_outputs * 2 * K + physical_av)
    assert dict(op.metadata)["requested_matrix_flops"] == requested
    assert work.matrix == {"bf16": physical}


@pytest.mark.parametrize("rows", (1, 15, 16, 17, 32, 33, 48, 63, 64))
def test_l0_operands_reload_per_value_head_while_gm_qk_is_reused(rows):
    # Exact-image ascend950/copy_l1_to_l0a.hpp:49-59 and l0b.hpp:60-73
    # ceil the two LoadData2D steps. One BF16 fractal is 16x16 elements.
    op = gdn_output.make_op(params(lengths=(rows,), heads=3, key_heads=1))
    plan = gdn_output.build(op, Geometry(4, {"vector": 8}))
    operand_tiles = ((BT, K), (K, BT), (BT, K), (K, V),
                     (rows, rows), (rows, V))
    expected = sum(
        16 * 16 * 2 for _ in range(3) for r, c in operand_tiles
        for _row in range(0, r, 16) for _col in range(0, c, 16))
    assert plan.work.resources["mte1"] == expected


@pytest.mark.parametrize("value_first", (False, True))
def test_output_addresses_and_numerical_state_layout_match_token_first_storage(value_first):
    rng = np.random.default_rng(711)
    rows, heads = 5, 3
    op = gdn_output.make_op(params(
        lengths=(rows,), heads=heads, key_heads=1, state_v_first=value_first))
    key_stride, value_stride = dict(op.metadata)["h_semantic_strides"]
    q, k = rng.normal(size=(2, rows, K)) / 8
    values = rng.normal(size=(heads, rows, V))
    states = rng.normal(size=(heads, K, V))
    gates = -rng.random((heads, rows))
    actual = np.empty(rows * heads * V)
    reference = np.zeros((rows, heads, V))
    addresses = []
    scale = K ** -0.5
    for head in range(heads):
        physical = np.ascontiguousarray(states[head].T if value_first else states[head])
        logical = physical.ravel()[
            np.arange(K)[:, None] * key_stride + np.arange(V)[None, :] * value_stride]
        qh = q @ logical
        qk = q @ k.T
        gate = np.exp2(gates[head, :, None] - gates[head, None, :])
        local = np.tril(qk * gate) @ values[head]
        output = scale * (np.exp2(gates[head])[:, None] * qh + local)
        for row in range(rows):
            previous = sum(q[row, channel] * states[head, channel] for channel in range(K))
            reference[row, head] = np.exp2(gates[head, row]) * previous
            for prior in range(row + 1):
                dot = sum(q[row, channel] * k[prior, channel] for channel in range(K))
                reference[row, head] += dot * np.exp2(
                    gates[head, row] - gates[head, prior]) * values[head, prior]
            reference[row, head] *= scale
            for channel in range(V):
                offset = gdn_output.output_offset(row, head, channel, heads)
                addresses.append(offset)
                actual[offset] = output[row, channel]
    assert sorted(addresses) == list(range(actual.size))
    np.testing.assert_allclose(actual.reshape(rows, heads, V), reference,
                               rtol=1e-12, atol=1e-12)


def execute(value, case, trace_limit=0):
    op = gdn_output.make_op(value)
    parsed = parse_case(case)
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(gdn_output.build)})
    execution = Execution(parsed, CostModel(parsed), recipes=recipes, trace_limit=trace_limit)
    prepared, recipe = execution.prepare(op)
    context = execution._context(prepared, 0, "none")
    ready = execution.engine.submit("ready")
    done = recipe.lower(context).start(execution.engine, "operator", ready)
    execution.engine.advance(until=(done,))
    return execution.engine, done.time


def phase_event(engine, chunk, name, phase):
    suffixes = (f":tile{chunk}:{name}:program:{phase}",
                f":tile{chunk}:{name}:{phase}")
    return next(row for row in engine.trace if row["task"].endswith(suffixes))


def test_ub_belongs_to_each_aiv_and_matrix_buffers_to_each_aic():
    sizes = {"ub": 248 * 1024, "l1": 352 * 1024,
             "l0a": 64 * 1024, "l0b": 64 * 1024, "l0c": 256 * 1024}
    value = params(lengths=(64,), heads=12, key_heads=4)
    engine, with_pools = execute(value, hardware(buffers=sizes), trace_limit=10000)
    _, without_pools = execute(value, hardware())
    assert with_pools == without_pools
    assert engine.summary()["peak_allocated_per_unit"] == {
        f"{pool}@r0:s0:unit{unit}": size for pool, size in sizes.items()
        for unit in range(8 if pool == "ub" else 4)
    }
    assert not engine.allocations and not engine.unit_allocated
    seen = {f"{resource}@r0:s0": set()
            for resource in ("matrix", "vector", "mte1", "fixpipe")}
    for event in engine.trace:
        for resource in seen.keys() & event["demand"].keys():
            if event["demand"][resource]:
                owners = event["placement"][resource]
                assert len(owners) == 1
                seen[resource].update(owners)
    assert seen == {
        f"{resource}@r0:s0": set(range(8 if resource == "vector" else 4))
        for resource in ("matrix", "vector", "mte1", "fixpipe")
    }


def test_single_group_does_not_borrow_idle_matrix_or_vector_units():
    durations = []
    for units in (1, 32):
        case = hardware(units=units)
        device = case["machine"]["device"]
        device["compute_peaks"]["bf16"] = units * 1e9
        device["rates"] = {"memory": 1e30, "vector": 2 * units * 1e9,
                           "mte1": units * 1e30, "fixpipe": units * 1e30}
        _, duration = execute(params(heads=1, key_heads=1), case)
        durations.append(duration)
    assert durations[1] == pytest.approx(durations[0], rel=1e-12)


def test_live_execution_window_does_not_grow_with_token_history():
    summaries = [
        execute(params(lengths=(length,)), hardware())[0].summary()
        for length in (256, 1024)
    ]
    short, long = summaries
    assert long["peak_live_tasks"] == short["peak_live_tasks"]
    assert long["completed_tasks"] > 3 * short["completed_tasks"]
    assert long["live_tasks"] == short["live_tasks"] == 0


def test_stage4_v_prefetch_uses_ordered_copy_service_and_overlaps_first_av():
    # cube.h:142-171 issues V0 then V1 on one MTE2 stream before AV0.
    # They must not share half the copy rate and finish simultaneously.
    case = hardware(units=1, split_copy=True)
    device = case["machine"]["device"]
    device["compute_peaks"]["bf16"] = 1e10
    device["rates"].update(memory=1e30, vector=1e30, read=1e6, write=1e30)
    engine, _ = execute(params(heads=4, key_heads=1), case, trace_limit=10000)
    first = phase_event(engine, 0, "v_load0", "load")
    lookahead = phase_event(engine, 0, "v_load1", "load")
    compute = phase_event(engine, 0, "av0", "compute")
    assert lookahead["start_ms"] >= first["end_ms"]
    assert compute["start_ms"] < lookahead["end_ms"]


@pytest.mark.parametrize("heads", (3, 4))
def test_next_group_loads_overlap_previous_outputs_but_compute_waits_both_aivs(heads):
    case = hardware(units=1, split_copy=True)
    case["machine"]["device"]["rates"].update(
        memory=1e30, vector=1e6, read=1e10, write=1e6)
    engine, _ = execute(params(lengths=(128,), heads=heads, key_heads=1),
                        case, trace_limit=10000)

    previous_outputs = [
        phase_event(engine, 0, f"output{head}", "store") for head in range(heads)
    ]
    both_aivs_done = max(row["end_ms"] for row in previous_outputs)
    for load_name in ("qk_load0", "h_load0"):
        copied = phase_event(engine, 1, load_name, "load")
        assert copied["start_ms"] < both_aivs_done
    assert phase_event(engine, 1, "qk0", "compute")["start_ms"] >= both_aivs_done


def test_qk_compute_overlaps_h_transfer_and_qh_waits_for_h():
    # ProcessStage2Head queues H after Q/K, but QK waits only qkEvent.
    # The H-ready fence belongs to QH, after the QK MMAD/publication.
    case = hardware(units=1, split_copy=True)
    case["machine"]["device"]["compute_peaks"]["bf16"] = 1e10
    case["machine"]["device"]["rates"].update(
        memory=1e30, vector=1e30, read=1e6, write=1e30)
    engine, _ = execute(params(heads=1, key_heads=1), case, trace_limit=10000)

    qk_loaded = phase_event(engine, 0, "qk_load0", "load")
    h_loaded = phase_event(engine, 0, "h_load0", "load")
    qk = phase_event(engine, 0, "qk0", "compute")
    qh = phase_event(engine, 0, "qh0", "compute")
    assert h_loaded["start_ms"] >= qk_loaded["end_ms"]
    assert qk["start_ms"] >= qk_loaded["end_ms"]
    assert qk["start_ms"] < h_loaded["end_ms"]
    assert qh["start_ms"] >= h_loaded["end_ms"]
