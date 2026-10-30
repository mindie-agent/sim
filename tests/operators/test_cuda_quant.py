"""Independent source execution, address and resource controls for CUDA quant."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.operators import cuda_quant
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from tools.run_operator_cases import control_case, run


def hardware(units=1, *, per_owner_rate=1e6, memory_rate=1e9, shared_bytes=48):
    raw = control_case(units=units, service_roles=cuda_quant.SERVICE_ROLES)
    device = raw["machine"]["device"]
    device["resource_units"].update(vector=units, shared_memory=units)
    device["rates"].update(memory=memory_rate, vector=units * per_owner_rate,
                           shared_memory=units * per_owner_rate)
    for mode in device["service_modes"].values():
        mode["rate"] = units * per_owner_rate
    device["buffers"] = {"smem": shared_bytes}
    device["buffer_units"] = {"smem": units}
    return raw


def execute(params, raw=None, *, ready=0):
    case = parse_case(raw or hardware())
    op = cuda_quant.make_op(params)
    recipes = RecipeBook(recipes={op.recipe: PlannedRecipe(cuda_quant.build)})
    execution = Execution(case, CostModel(case), recipes=recipes, trace_limit=20000)
    op, recipe = execution.prepare(op)
    context = execution._context(op, 0, "none")
    engine = execution.engine
    complete = recipe.lower(context).start(engine, "quant", engine.submit("ready", release=ready))
    return engine, complete


def test_all_256_cub_sizes_match_independent_predicated_execution():
    for block in range(1, 257):
        values = np.random.default_rng(block).normal(size=block)
        counts, steps, combine = cub_execution(block, values)
        inventory = cuda_quant.cub_inventory(block)
        assert inventory["max_active_lanes"] == counts.get("max_calls", 0)
        assert inventory["shuffle_active_lanes"] == counts["shuffle_active_lanes"]
        assert inventory["warp_shuffle_calls"] == counts["shuffle_warp_calls"]
        assert inventory["thread0_max_calls"] == len(combine)
        phase_work = dict(cuda_quant.source_phases(block, "fp32"))
        for offset in (1, 2, 4, 8, 16):
            active = sum(len(step["max_lanes"]) for step in steps if step["offset"] == offset)
            assert phase_work[f"warp_max_{offset}"].resources.get("fp32_max", 0) == active


def test_storage_offsets_follow_real_scalar_pack_and_tail_paths():
    for dtype, size in (("bf16", 2), ("fp16", 2), ("fp32", 4)):
        for rows, width in ((1, 1), (5, 17), (1, 32), (19, 33), (17, 273)):
            for input_base, output_base in ((0, 0), (size, 0), (size, 1), (7 * size, 9)):
                op = cuda_quant.make_op({
                    "rows": rows, "width": width, "dtype": dtype,
                    "input_byte_offset": input_base, "output_byte_offset": output_base})
                summary = dict(op.metadata)["traversal"]
                for output, name in ((False, "read"), (True, "quant")):
                    counts = Counter()
                    for row in range(rows):
                        seen, _, fast, scalar = traversal(
                            width, row, size, input_base=input_base,
                            output_base=output_base, output=output)
                        counts.update(seen)
                        counts["fast_rows"] += fast
                        counts["scalar_rows"] += scalar
                    assert summary[name] == {
                        "fast_rows": counts["fast_rows"], "scalar_rows": counts["scalar_rows"],
                        "prefix_elements": counts["prefix"], "tail_elements": counts["tail"],
                        "vector_packs": counts["pack"], "vector_elements": 16 * counts["pack"],
                    }
    offset = cuda_quant.traversal_summary(1, 32, "bf16", 2, 0)
    assert offset["quant"]["scalar_rows"] == 1
    assert offset["quant"]["tail_elements"] == 32
    assert offset["quant"]["vector_packs"] == 0


def test_scale_and_payload_stores_wait_for_their_actual_producers():
    engine, complete = execute({"rows": 1, "width": 32, "dtype": "fp32"},
                               hardware(per_owner_rate=100000, memory_rate=1e6))
    engine.advance(until=(complete,))
    trace = {row["task"].split(":")[-1]: row for row in engine.trace}
    for producer, consumer in (
        ("read_absmax_input", "absolute"), ("warp_max_16", "share_warp_max"),
        ("publish_absmax", "compute_scale"), ("compute_scale", "store_scale"),
        ("store_scale", "read_inverse_predicate"), ("read_quant_input", "multiply_inverse"),
        ("multiply_inverse", "convert_int8"), ("convert_int8", "store_int8"),
    ):
        assert trace[consumer]["start_ms"] >= trace[producer]["end_ms"]
    # Source phases use serial producers. At this instant the slow max is
    # still executing; only the first128B global input load may have serviced.
    engine, _ = execute({"rows": 1, "width": 32, "dtype": "fp32"},
                        hardware(per_owner_rate=100000, memory_rate=1e6))
    engine.advance(horizon=.65)
    assert engine.serviced["memory@r0:s0"] == 128


def test_typed_service_rates_change_only_the_declared_instruction_class():
    params = {"rows": 1, "width": 32, "dtype": "bf16"}
    raw = hardware()
    op = cuda_quant.make_op(params)
    base = verify_operator(raw, op, cuda_quant.build)
    raw["machine"]["device"]["service_modes"]["cvt_bf16_fp32"]["rate"] /= 4
    slow = verify_operator(raw, op, cuda_quant.build)
    assert base["status"] == slow["status"] == "passed"
    #64 source conversion lanes at1000/ms vs250/ms; other source work is identical.
    assert slow["result"]["duration_ms"] - base["result"]["duration_ms"] == pytest.approx(.192)
    assert slow["result"]["operator"] == base["result"]["operator"]
    del raw["machine"]["device"]["service_modes"]["cvt_bf16_fp32"]
    with pytest.raises(ValueError, match="physical_resource_rate_missing:cvt_bf16_fp32"):
        verify_operator(raw, op, cuda_quant.build)


def test_zero_row_branch_and_input_dtype_have_distinct_source_work():
    for dtype in ("bf16", "fp16", "fp32"):
        params = {"rows": 3, "width": 33, "dtype": dtype}
        normal = cuda_quant.make_op(params)
        zero = cuda_quant.make_op({**params, "zero_absmax": True})
        normal_work = dict(normal.metadata)["resource_work"]
        zero_work = dict(zero.metadata)["resource_work"]
        assert normal_work["fp32_div"] == 3 * 34
        assert zero_work["fp32_div"] == 3
        for work in (normal_work, zero_work):
            if dtype != "fp32":
                assert work[f"cvt_{dtype}_fp32"] == 2 * 3 * 33
            assert work["cvt_fp32_int8"] == 3 * 33
        for op in (normal, zero):
            assert verify_operator(hardware(), op, cuda_quant.build)["status"] == "passed"


def test_owned_ctas_share_only_global_memory_and_recycle_shared_storage():
    for units in (1, 4, 32):
        one = cuda_quant.make_op({"rows": 1, "width": 32, "dtype": "fp32"})
        many = cuda_quant.make_op({"rows": units, "width": 32, "dtype": "fp32"})
        raw = hardware(units, memory_rate=1e30)
        single = verify_operator(raw, one, cuda_quant.build, input_ready_ms=7)
        parallel = verify_operator(raw, many, cuda_quant.build, input_ready_ms=7)
        assert single["status"] == parallel["status"] == "passed"
        assert single["result"]["duration_ms"] == pytest.approx(.627)
        assert parallel["result"]["duration_ms"] == pytest.approx(single["result"]["duration_ms"])
        assert parallel["result"]["peak_buffers_per_unit"] == {
            f"smem@r0:s0:unit{owner}": 48 for owner in range(units)}
        memory_bound = verify_operator(hardware(units, per_owner_rate=1e30, memory_rate=1e6),
                                       many, cuda_quant.build)
        assert memory_bound["result"]["duration_ms"] == pytest.approx(units * .292)
        assert not any(memory_bound["result"]["remaining_buffers"].values())
    with pytest.raises(ValueError, match="shared_memory_capacity_exceeded"):
        verify_operator(hardware(shared_bytes=47), one, cuda_quant.build)


def test_fixed_cases_and_large_row_counts_keep_the_active_window_bounded():
    cases = json.loads((Path(__file__).resolve().parents[2]
                        / "tests/fixtures/operator_cases/cuda_quant.json").read_text())["cases"]
    report = run(cuda_quant, cases)
    assert report["status"] == "passed"
    for rows in (100000, 1000000):
        engine, complete = execute({"rows": rows, "width": 5120, "dtype": "bf16"}, hardware(32))
        engine.advance(horizon=0)
        assert not complete.done
        assert engine.summary()["live_tasks"] <= 64 * 32
        assert engine.summary()["live_programs"] <= 35


def cub_execution(block, values=None):
    # Execute generic float ReduceStep at offsets 1,2,4,8,16. The template
    # has 256 threads, but only blockDim.x physical threads invoke the function.
    lanes = list(map(float, values if values is not None else range(block)))
    counter = Counter()
    steps = []
    aggregates = []
    for first in range(0, block, 32):
        warp = lanes[first:first+32]
        last_lane = len(warp)-1
        for offset in (1,2,4,8,16):
            before = list(warp)
            live = []
            for lane in range(len(warp)):
                counter['shuffle_active_lanes'] += 1
                if offset+lane <= last_lane:
                    counter['max_calls'] += 1
                    live.append(lane)
                    warp[lane] = max(before[lane], before[lane+offset])
            counter['shuffle_warp_calls'] += 1
            counter['max_nonempty_warp_steps'] += bool(live)
            steps.append({'warp':first//32,'offset':offset,'max_lanes':live})
        aggregates.append(warp[0])
        counter['shared_aggregate_stores'] += 1
    total = aggregates[0]
    combine = []
    # Successors are tested against num_valid for FULL_TILE=false; this loop
    # enumerates all seven compile-time candidate successors independently.
    for successor in range(1,8):
        if block >= 256 or successor*32 < block:
            counter['thread0_shared_reads'] += 1
            counter['thread0_max_calls'] += 1
            total = max(total, aggregates[successor])
            combine.append(successor)
    assert total == max(lanes)
    return dict(counter), steps, combine


def traversal(width, row, size, *, input_base=0, output_base=0, output=False):
    threads = min(width,256)
    in_address = input_base + row*width*size
    out_address = output_base + row*width
    pack_bytes = 16*size
    events = []
    fast = in_address % pack_bytes == 0 and width % 16 == 0
    if output:
        fast = fast and out_address % 16 == 0
    prefix = 0
    while prefix < width and prefix < 16 and (in_address+prefix*size) % pack_bytes:
        prefix += 1
    scalar_all = output and (out_address+prefix) % 16 != 0 and not fast
    for thread in range(threads):
        if scalar_all:
            for i in range(thread,width,threads): events.append(('tail',thread,(i,)))
            continue
        if not fast:
            for i in range(thread,prefix,threads): events.append(('prefix',thread,(i,)))
        remaining = width-prefix
        for group in range(thread,remaining//16,threads):
            events.append(('pack',thread,tuple(prefix+group*16+j for j in range(16))))
        for i in range(thread+(remaining//16)*16,remaining,threads):
            events.append(('tail',thread,(prefix+i,)))
    seen = sorted(i for _,_,indices in events for i in indices)
    assert seen == list(range(width))
    counts = Counter(kind for kind,_,_ in events)
    return dict(counts), events, fast, scalar_all


def f32(value):
    """Cast through an IEEE-754 binary32 slot without importing torch."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def bf16_round(values):
    """Round binary32 values to nearest bfloat16, ties to even."""
    array = np.asarray(values, dtype=np.float32)
    bits = array.view(np.uint32).copy()
    lsb = (bits >> 16) & np.uint32(1)
    rounded = (bits + np.uint32(0x7FFF) + lsb) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def as_input(values, dtype):
    array = np.asarray(values, dtype=np.float32)
    if dtype == "bf16":
        return bf16_round(array)
    if dtype == "fp16":
        return np.asarray(array, dtype=np.float16).astype(np.float32)
    if dtype == "fp32":
        return array.copy()
    raise ValueError(dtype)


def python_rint_sat(value):
    """Round-to-nearest-even then clamp, using Python's scalar round."""
    rounded = int(round(float(np.float32(value))))
    return max(-128, min(127, rounded))


def rn_sat_int8(values):
    """cvt.rni.sat.s8.f32: round-to-nearest-even, then saturate to int8."""
    rounded = np.rint(np.asarray(values, dtype=np.float32))
    clamped = np.clip(rounded, np.float32(-128.0), np.float32(127.0))
    return clamped.astype(np.int8)


def cpu_kernel_quant(values, dtype):
    """Independent NumPy model of dynamic_scaled_int8_quant_kernel.

    The inverse is recomputed from ``absmax`` exactly as the kernel does after
    the block barrier.  It is deliberately *not* obtained from the stored
    ``scale`` tensor.
    """
    array = as_input(values, dtype)
    absmax = np.max(np.abs(array), axis=1).astype(np.float32)
    scale = np.divide(absmax, np.float32(127.0), dtype=np.float32)
    inverse = np.zeros_like(absmax, dtype=np.float32)
    nonzero = absmax != 0
    inverse[nonzero] = np.divide(np.float32(127.0), absmax[nonzero],
                                 dtype=np.float32)
    scaled = np.multiply(array, inverse[:, None], dtype=np.float32)
    return rn_sat_int8(scaled), scale, inverse


def python_kernel_quant(values, dtype):
    """Scalar binary32 oracle independent of the NumPy vector path."""
    array = as_input(values, dtype)
    rows = array.shape[0]
    quantized, scales = [], []
    for row in range(rows):
        prepared = [f32(value) for value in array[row]]
        absmax = max(abs(value) for value in prepared)
        scale = f32(f32(absmax) / f32(127.0))
        inverse = f32(0.0) if absmax == 0 else f32(f32(127.0) / f32(absmax))
        output = [python_rint_sat(f32(f32(value) * inverse)) for value in prepared]
        quantized.append(output)
        scales.append(scale)
    return np.asarray(quantized, dtype=np.int8), np.asarray(scales, dtype=np.float32)


def test_cpu_rint_saturation_is_round_to_nearest_even_then_clamped():
    values = np.asarray(
        [127.5, 128.5, -128.5, -129.5, 2.5, 3.5, 0.5, -0.5, -2.5,
         126.5, -126.5, 127.49999, 127.50001, 200.0, -200.0],
        dtype=np.float32,
    )
    assert rn_sat_int8(values).tolist() == [
        127, 127, -128, -128, 2, 4, 0, 0, -2, 126, -126, 127, 127, 127, -128]
    assert [python_rint_sat(value) for value in values] == [
        127, 127, -128, -128, 2, 4, 0, 0, -2, 126, -126, 127, 127, 127, -128]
    # The source instruction rounds before it saturates: 127.5 would be 128,
    # then the int8 saturation clamps it to 127.  -128.5 rounds to -128.
    assert np.rint(np.float32(127.5)) == np.float32(128.0)
    assert np.rint(np.float32(-128.5)) == np.float32(-128.0)


def test_cpu_oracle_recomputes_inverse_from_absmax_not_from_stored_scale():
    # Both operands are exact bfloat16 values.  Source: inv = 127.f / 3.625f
    # gives 35.034481048583984, whose product with 1.8125 rounds down to 63.
    # The stored scale alone gives 1.0f / scale = 35.03448486328125, whose
    # product is just above 63.5 and rounds to 64.
    values = np.asarray([[3.625, 1.8125]], dtype=np.float32)
    quantized, scale, inverse = cpu_kernel_quant(values, "bf16")
    assert scale[0] == np.float32(3.625) / np.float32(127.0)
    assert inverse[0] == np.float32(127.0) / np.float32(3.625)
    assert quantized.tolist() == [[127, 63]]
    alternate_inverse = np.float32(1.0) / scale[0]
    assert alternate_inverse != inverse[0]
    assert rn_sat_int8(np.float32(1.8125) * alternate_inverse).item() == 64
    stored_scale_reference = rn_sat_int8(
        np.divide(values, scale[:, None], dtype=np.float32)
    )
    assert stored_scale_reference.tolist() == [[127, 64]]


def test_cpu_oracle_zero_row_and_negative_maximum():
    values = np.zeros((1, 5), dtype=np.float32)
    quantized, scale, inverse = cpu_kernel_quant(values, "bf16")
    assert scale.tolist() == [0.0]
    assert inverse.tolist() == [0.0]
    assert quantized.tolist() == [[0, 0, 0, 0, 0]]
    negative = np.asarray([[-3.625, -1.8125]], dtype=np.float32)
    quantized, _, inverse = cpu_kernel_quant(negative, "bf16")
    assert inverse[0] == np.float32(127.0) / np.float32(3.625)
    assert quantized.tolist() == [[-127, -63]]


def test_cpu_oracle_matches_scalar_binary32_oracle_for_each_input_dtype():
    rng = np.random.default_rng(20261028)
    for dtype in ("bf16", "fp16", "fp32"):
        for rows, width, scale in ((6, 37, 250.0), (4, 273, 1000.0)):
            values = rng.normal(0.0, scale, size=(rows, width)).astype(np.float32)
            values[1] = 0.0
            numpy_q, numpy_scale, numpy_inverse = cpu_kernel_quant(values, dtype)
            scalar_q, scalar_scale = python_kernel_quant(values, dtype)
            assert np.array_equal(numpy_q, scalar_q), (dtype, rows, width)
            assert np.array_equal(numpy_scale.view(np.uint32),
                                  scalar_scale.view(np.uint32)), dtype
            assert numpy_scale.shape == (rows,)
            assert numpy_inverse.shape == (rows,)
