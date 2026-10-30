"""Independent checks for the source-derived causal-conv prefill plan.

The plan is not compared with a captured duration.  These tests establish:
  * the selected fwd dispatch, the literal wrapper-default null-block branch,
    the explicit no-null branch, and rejected update/APC/unsupported paths;
  * an explicit source-line work walk over every ``(chunk, channel-block)``
    program, including conditional barriers and null-block early returns;
  * physical CTA ownership: one CTA's vector service is one SM's share and does
    not become 32x faster when 1 -> 32 SMs are exposed at a fixed per-SM rate;
  * lazy per-owner CTA streams over the source grid, with no global two-slot
    window;
  * independent NumPy rolling-convolution and source-tiled oracles covering
    initial/final state, chunk boundaries, channel tails, token tails, bias,
    SiLU and null-block sequences;
  * the coordinator's unchanged wave-3 cases, development and heldout.
"""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from mindie_sim.prediction.mechanism import Geometry
from mindie_sim.prediction.program import (
    ParallelProgram,
    PipelineProgram,
    RecipeContext,
)
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.prediction.operators import causal_conv_prefill as conv
from mindie_sim.schema.common import ModelingError
from tools.run_operator_cases import control_case, run as run_cases


CASES = Path(__file__).parents[2] / "tests/fixtures/operator_cases/causal_conv_prefill.json"
# Source launch constants, written independently of the module under test.
TILE_M = 8
TILE_N = 256


def base_params(**changes):
    params = {
        "lengths": [3],
        "channels": 128,
        "width": 4,
        "dtype": "bf16",
        "initial_state": False,
        "bias": False,
        "activation": "silu",
        "implementation": "triton_causal_conv1d_fwd",
        "is_apc_enabled": False,
        "has_null_block": False,
    }
    params.update(changes)
    return params


def geometry(compute_units=32, vector_units=None):
    if vector_units is None:
        vector_units = 2 * compute_units
    return Geometry(compute_units=compute_units,
                    resource_units={"vector": vector_units},
                    buffers={}, buffer_units={})


def recipe_context(op, compute_units=32, vector_units=None):
    if vector_units is None:
        vector_units = 2 * compute_units
    return RecipeContext(
        op=op, work={},
        resource_names={"hbm": "memory@r0:s0", "vector": "vector@r0:s0"},
        rates={"hbm": 1e12, "vector": 1e12}, core_count=compute_units,
        resource_units={"hbm": compute_units, "vector": vector_units},
        pool_units={},
    )


def inspect_program(op, compute_units=32, vector_units=None):
    return conv.build(op, geometry(compute_units, vector_units)).program(
        recipe_context(op, compute_units, vector_units))


def iter_stages(program):
    """Walk a small program's lazy tiles; tests only use small grids here."""
    if isinstance(program, ParallelProgram):
        for child in program.programs:
            yield from iter_stages(child)
    elif isinstance(program, PipelineProgram):
        for index in range(program.count):
            yield from program.tile(index)
    else:
        for stage in program.stages:
            yield stage


def stage_work(program, role):
    name = {"hbm": "memory@r0:s0", "vector": "vector@r0:s0"}[role]
    return sum(stage.work.get(name, 0.0) for stage in iter_stages(program))


def hardware_case(units, *, per_sm_memory=1e15, per_sm_vector=1e12):
    """Synthetic per-SM rates; total rates scale with the SM count."""
    case = control_case(units=units)
    case["machine"]["device"]["rates"] = {
        "memory": units * per_sm_memory, "vector": units * per_sm_vector}
    return case


def skipped_flags(op):
    meta = dict(op.metadata)
    if not meta["has_null_block"]:
        return [False] * len(meta["lengths"])
    return [value == meta["null_block_id"] for value in meta["cache_indices"]]


def explicit_source_work(lengths, channels, width, initial, bias, activation,
                         skipped=None, element_bytes=2.0):
    """Walk every Triton program with line-referenced work, independently."""
    state_len = width - 1
    np2 = 1 << (state_len - 1).bit_length()
    per_token = 1 + 2 * width + (width - 1) + (4 if activation else 0) + 1
    skipped = list(skipped or [False] * len(lengths))
    hbm = 0.0
    vector = 0.0
    barriers = 0
    for length, has_initial, is_skipped in zip(lengths, initial, skipped):
        chunks = (length + TILE_M - 1) // TILE_M
        for chunk_offset in range(chunks):
            segment = min(TILE_M, length - chunk_offset * TILE_M)
            for feature_block in range((channels + TILE_N - 1) // TILE_N):
                active = min(TILE_N, channels - feature_block * TILE_N)
                if is_skipped:
                    # Lines 75-76, 84-85, 132-139: scalar loads then return.
                    hbm += 20.0
                    continue
                first = chunk_offset == 0

                # Lines 397-409 and 386-393: every program preloads weights and
                # optional bias.  Lines 151-181: chunk 0 may read prior state.
                hbm += active * width * element_bytes
                if bias:
                    hbm += active * element_bytes
                if first and has_initial:
                    hbm += state_len * active * element_bytes
                # Lines 75-76, 84-85, 132-134: 20 scalar bytes per program.
                hbm += 20.0
                if first:
                    hbm += 1.0  # has_initial_states_ptr, line 153

                # Lines 411-465: one x load per token, one output store.
                hbm += active * segment * element_bytes
                hbm += active * segment * element_bytes
                if not first:
                    # Lines 309-338: KERNEL_WIDTH-1 prior x tokens.
                    hbm += (width - 1) * active * element_bytes

                if first:
                    if state_len <= length:
                        # Lines 202-235: last state_len x tokens and the output
                        # cache-index load at line 219.
                        hbm += state_len * active * element_bytes
                        hbm += 4.0
                    else:
                        # Lines 237-307: shifted old state plus first x tokens.
                        if has_initial:
                            hbm += (state_len - length) * active * element_bytes
                        hbm += length * active * element_bytes
                    # Lines 225-235/273-280/300-307: state_len values stored.
                    hbm += state_len * active * element_bytes
                    if state_len > length and has_initial:
                        # Lines 269-272: tl.where over NP2_STATELEN x BLOCK_N.
                        vector += float(TILE_N * np2)
                    # Lines 234/269: only S <= L or the shifting branch barriers.
                    if state_len <= length or has_initial:
                        barriers += 1

                vector += float(TILE_N * segment * per_token)
                if bias:
                    # acc_preload conversion, lines 389-392.
                    vector += float(TILE_N)
    return hbm, vector, barriers


def test_selected_fwd_dispatch_and_unsupported_paths_are_explicit():
    op = conv.make_op(base_params())
    dispatch = dict(op.metadata)["dispatch"]
    assert op.recipe == "triton_causal_conv1d_fwd"
    assert dispatch["implementation"] == "triton_causal_conv1d_fwd"
    assert dispatch["is_apc_enabled"] is False
    assert dispatch["has_null_block"] is False
    assert dispatch["kernel_function"] == "_causal_conv1d_fwd_kernel"
    assert dispatch["state_len"] == 3 and dispatch["np2_statelen"] == 4
    assert dispatch["block_m"] == TILE_M and dispatch["block_n"] == TILE_N
    assert dispatch["num_stages"] == 2 and dispatch["cta_window"] >= 1
    assert op.returns == ("final_conv_state",)
    meta = dict(op.metadata)
    assert meta["source"]["sha256"] == conv.SOURCE_SHA256
    assert meta["source"]["bytes"] == conv.SOURCE_BYTES
    assert meta["null_block_source"]["sha256"] == conv.NULL_BLOCK_SOURCE_SHA256
    assert meta["null_block_source"]["line"] == 46

    failures = (
        ({"implementation": "triton_causal_conv1d_update"}, "unsupported_implementation"),
        ({"is_apc_enabled": True}, "apc_not_selected"),
        ({"width": 1}, "kernel_width_unsupported"),
        ({"width": 5}, "kernel_width_unsupported"),
        ({"dtype": "fp8"}, "dtype_unsupported"),
        ({"channels": 0}, "channels_must_be_positive"),
        ({"lengths": [3, -1]}, "lengths_must"),
        ({"lengths": [3, 4], "initial_state": [True]}, "initial_state_length_or_type_mismatch"),
        ({"unknown": 1}, "unknown_parameter"),
    )
    for changes, match in failures:
        with pytest.raises(ModelingError, match=match):
            conv.make_op(base_params(**changes))

    # Supplying APC and null-block false explicitly is the selected path; a
    # different source branch is never treated as a fallback.
    per_sequence = conv.make_op(base_params(
        lengths=[0, 3], initial_state=[True, False], channels=130,
        dtype="bfloat16",
    ))
    meta = dict(per_sequence.metadata)
    assert meta["dtype"] == "bf16"
    assert meta["has_initial_state"] == (True, False)
    assert meta["counts"]["programs"] == 1
    assert meta["counts"]["state_write_programs"] == 1
    assert meta["counts"]["barrier_instructions"] == 1


def test_literal_wrapper_default_null_block_is_accepted_and_models_early_return():
    # The source wrapper default is null_block_id=0 (utils.py:46), so the
    # literal default call selects HAS_NULL_BLOCK=True (line 738).
    params = base_params()
    params.pop("has_null_block")
    op = conv.make_op(params)
    meta = dict(op.metadata)
    assert meta["dispatch"]["has_null_block"] is True
    assert meta["dispatch"]["null_block_id"] == 0
    assert meta["cache_indices"] == (0,)
    # Identity cache line 0 is the reserved null block: every CTA returns after
    # the 20 scalar bytes and the final state is not rewritten.
    assert meta["counts"]["early_return_programs"] == 1
    assert meta["counts"]["processed_programs"] == 0
    assert meta["counts"]["state_write_programs"] == 0
    assert meta["counts"]["barrier_instructions"] == 0
    work = conv.build(op, geometry()).work.resources
    assert work == {"hbm": 20.0, "vector": 0.0}

    # The identity default skips only the sequence whose line is null.
    multi_params = base_params(lengths=[3, 4], channels=130)
    multi_params.pop("has_null_block")
    multi = conv.make_op(multi_params)
    multi_counts = dict(multi.metadata)["counts"]
    assert multi_counts["early_return_programs"] == 1
    assert multi_counts["processed_programs"] == 1
    assert multi_counts["state_write_programs"] == 1

    # The explicit no-null caller flag remains a distinct, supported branch.
    explicit = conv.make_op(base_params())
    assert dict(explicit.metadata)["dispatch"]["has_null_block"] is False
    assert dict(explicit.metadata)["counts"]["state_write_programs"] == 1

    # A valid non-null cache line exercises the processed path with the flag on.
    valid = conv.make_op(base_params(
        has_null_block=True, cache_indices=[1], lengths=[3]))
    valid_meta = dict(valid.metadata)
    assert valid_meta["counts"]["early_return_programs"] == 0
    assert valid_meta["counts"]["state_write_programs"] == 1

    # Explicit null_block_id=None is the source's no-null specialization.
    no_null = conv.make_op(base_params(
        has_null_block=False, null_block_id=None, cache_indices=[0]))
    assert dict(no_null.metadata)["dispatch"]["has_null_block"] is False
    assert dict(no_null.metadata)["counts"]["state_write_programs"] == 1

    with pytest.raises(ModelingError, match="null_block_id_required"):
        conv.make_op(base_params(has_null_block=True, null_block_id=None))


def test_null_block_early_return_conserves_scalar_work_and_skips_state():
    op = conv.make_op(base_params(
        lengths=[3, 4], channels=130, initial_state=[True, True],
        bias=True, has_null_block=True, cache_indices=[0, 1],
    ))
    meta = dict(op.metadata)
    counts = meta["counts"]
    assert counts["programs"] == 2
    assert counts["processed_programs"] == 1
    assert counts["early_return_programs"] == 1
    assert counts["state_write_programs"] == 1
    assert counts["barrier_instructions"] == 1

    plan = conv.build(op, geometry(2))
    expected = explicit_source_work(
        [3, 4], 130, 4, [True, True], True, "silu", [True, False])
    assert plan.work.resources["hbm"] == pytest.approx(expected[0])
    assert plan.work.resources["vector"] == pytest.approx(expected[1])
    assert counts["barrier_instructions"] == expected[2]

    program = inspect_program(op, compute_units=2)
    assert isinstance(program, ParallelProgram)
    names = []
    for stream in program.programs:
        for index in range(stream.count):
            names.append([stage.name for stage in stream.tile(index)])
    assert ["scalar"] in names
    processed = next(stages for stages in names if stages != ["scalar"])
    assert processed[0] == "prep" and "state_rewrite" not in processed
    assert "barrier" in processed
    assert stage_work(program, "hbm") == pytest.approx(plan.work.resources["hbm"])
    assert stage_work(program, "vector") == pytest.approx(plan.work.resources["vector"])


@pytest.mark.parametrize("lengths,channels,width,initial,bias,activation,cache_indices", [
    ([3, 127], 130, 4, [True, False], True, "silu", None),
    ([1, 65], 256, 4, [False, True], False, "silu", None),
    ([64, 1], 128, 4, [True, True], False, "silu", None),
    ([9, 0], 257, 4, [True, False], True, "silu", None),
    ([5, 5], 7, 3, [True, True], True, None, None),
    ([5], 16, 2, [True], True, "silu", None),
    ([2], 130, 4, [False], True, "silu", None),      # S > L, no prior state
    ([9], 130, 4, [False], False, "silu", None),     # S <= L, no prior state
    ([8, 9, 16], 257, 4, [True, False, True], True, "silu", None),
    ([2], 257, 4, [True], True, "silu", None),  # S > L, two channel blocks
    ([3, 4], 130, 4, [True, True], True, "silu", [0, 1]),  # null seq 0
    ([3, 4], 257, 4, [True, True], True, "silu", [0, 1]),  # null + two blocks
])
def test_plan_work_equals_an_independent_explicit_program_walk(
        lengths, channels, width, initial, bias, activation, cache_indices):
    changes = {}
    if cache_indices is not None:
        changes = {"has_null_block": True, "cache_indices": cache_indices}
    op = conv.make_op(base_params(
        lengths=lengths, channels=channels, width=width,
        initial_state=initial, bias=bias, activation=activation, **changes,
    ))
    plan = conv.build(op, geometry())
    skipped = skipped_flags(op)
    expected_hbm, expected_vector, expected_barriers = explicit_source_work(
        lengths, channels, width, initial, bias, activation, skipped)
    assert plan.work.matrix == {}
    assert plan.work.resources["hbm"] == pytest.approx(expected_hbm)
    assert plan.work.resources["vector"] == pytest.approx(expected_vector)
    assert dict(op.metadata)["counts"]["barrier_instructions"] == expected_barriers
    assert set(plan.work.resources) == {"hbm", "vector"}
    # The per-owner streams partition the source grid and conserve its work.
    program = inspect_program(op, compute_units=4)
    assert sum(stream.count for stream in program.programs) == dict(
        op.metadata)["counts"]["programs"]
    assert stage_work(program, "hbm") == pytest.approx(expected_hbm)
    assert stage_work(program, "vector") == pytest.approx(expected_vector)


def test_contract_public_path_services_exactly_the_declared_work():
    op = conv.make_op(base_params(
        lengths=[3, 127], channels=130, initial_state=[True, False], bias=True,
    ))
    plan = conv.build(op, geometry())
    result = verify_operator(control_case(), op, conv.build)
    row = result["result"]
    assert result["status"] == "passed"
    assert result["finite_completion"] and result["buffers_released"]
    assert result["work_mismatches"] == {}
    assert row["declared_service"] == row["serviced"] == {
        "memory@r0:s0": plan.work.resources["hbm"],
        "vector@r0:s0": plan.work.resources["vector"],
    }
    assert row["operator"]["flops"] == 0.0
    assert row["operator"]["hbm_bytes"] == plan.work.resources["hbm"]
    assert row["operator"]["vector_ops"] == plan.work.resources["vector"]


def test_identity_does_not_change_work_or_completion():
    op = conv.make_op(base_params(lengths=[3, 127], channels=130, initial_state=True))
    renamed = replace(op, op_id="other", layer=97, requests=("unseen",))
    assert conv.build(op, geometry()).work == conv.build(renamed, geometry()).work
    first = verify_operator(control_case(), op, conv.build)["result"]
    second = verify_operator(control_case(), renamed, conv.build)["result"]
    assert first["duration_ms"] == second["duration_ms"]
    assert first["serviced"] == second["serviced"]


def test_element_bytes_scale_payload_but_not_padded_lane_work():
    common = dict(lengths=[9], channels=129, width=3, initial_state=False,
                  bias=False)
    bf16 = conv.build(conv.make_op(base_params(dtype="bf16", **common)), geometry())
    fp32 = conv.build(conv.make_op(base_params(dtype="fp32", **common)), geometry())
    # Scalar index traffic does not scale with the element dtype: 20 bytes for
    # each of two programs, one has_initial flag byte, and one 4-byte coord.
    scalar_bytes = 20 * 2 + 1 + 4
    assert (fp32.work.resources["hbm"] ==
            2 * (bf16.work.resources["hbm"] - scalar_bytes) + scalar_bytes)
    # bf16 stores convert fp32 accumulators; fp32 stores do not.  One extra
    # conversion per padded lane per token: 256 lanes * 9 tokens.
    assert bf16.work.resources["vector"] - fp32.work.resources["vector"] == 256 * 9


def test_initial_state_flag_does_not_suppress_final_state_write():
    off = conv.make_op(base_params(lengths=[65], channels=256, initial_state=False))
    on = conv.make_op(base_params(lengths=[65], channels=256, initial_state=True))
    off_work = conv.build(off, geometry()).work.resources
    on_work = conv.build(on, geometry()).work.resources
    # Both branches store the final length-3 state for the only sequence.
    assert off_work["vector"] == on_work["vector"]
    # The enabled branch additionally reads the whole prior state once.
    assert on_work["hbm"] - off_work["hbm"] == 3 * 256 * 2
    meta = dict(off.metadata)["counts"]
    assert meta["state_write_programs"] == 1


def test_single_cta_does_not_get_faster_with_more_sms_when_per_sm_rate_fixed():
    op = conv.make_op(base_params(lengths=[1], channels=128))
    durations = {}
    for units in (1, 32):
        result = verify_operator(hardware_case(units), op, conv.build)
        assert result["status"] == "passed"
        durations[units] = result["result"]["duration_ms"]
    # 4352 vector ops at 1e12 ops/s per SM = 4.352e-6 ms; scalar memory stages
    # are nonbinding.  One CTA owns one SM's share in both geometries.
    assert durations[1] == pytest.approx(4.352e-6, rel=2e-3)
    assert durations[32] == pytest.approx(durations[1], rel=2e-3)
    # The old unpinned plan completed a single CTA at the whole-device rate,
    # i.e. about 32x faster at 32 SMs.  This must not happen.
    assert durations[32] > durations[1] / 2

    program = inspect_program(op, compute_units=32)
    assert len(program.programs) == 1
    owner_units = program.programs[0].tile(0)
    compute = next(stage for stage in owner_units if stage.name == "conv_compute")
    assert compute.placement["vector@r0:s0"] == (0, 1)


def test_single_cta_consumes_one_sm_share_when_total_rate_is_fixed():
    op = conv.make_op(base_params(lengths=[1], channels=128))
    # Total device rate stays 1e12; the per-SM share falls as SMs are added.
    case = control_case(units=1)
    case["machine"]["device"]["rates"] = {"memory": 1e15, "vector": 1e12}
    one = verify_operator(case, op, conv.build)["result"]["duration_ms"]
    case = control_case(units=32)
    case["machine"]["device"]["rates"] = {"memory": 1e15, "vector": 1e12}
    many = verify_operator(case, op, conv.build)["result"]["duration_ms"]
    assert one == pytest.approx(4.352e-6, rel=2e-3)
    assert many / one == pytest.approx(32.0, rel=1e-2)
    assert many > one * 20  # never 32x faster with more SMs


def test_source_grid_order_and_per_owner_round_robin_streams():
    op = conv.make_op(base_params(
        lengths=[1, 9], channels=256, initial_state=True))
    # 3 flat chunks x 1 feature block; x is sequence-major chunk.
    program = inspect_program(op, compute_units=2)
    assert len(program.programs) == 2
    assert [stream.count for stream in program.programs] == [2, 1]

    stream0, stream1 = program.programs
    tiles0 = [stream0.tile(0), stream0.tile(1)]
    tile1 = stream1.tile(0)
    # global 0 -> sequence 0 chunk 0 (L=1); global 2 -> sequence 1 chunk 1.
    assert next(s for s in tiles0[0] if s.name == "conv_compute").work[
        "vector@r0:s0"] == 256 * 1 * 17
    assert next(s for s in tiles0[1] if s.name == "conv_compute").work[
        "vector@r0:s0"] == 256 * 1 * 17
    assert "state_rewrite" in [s.name for s in tiles0[0]]
    assert "state_read" not in [s.name for s in tiles0[1]]
    # global 1 -> sequence 1 chunk 0 (L=9); the segment is BLOCK_M=8 tokens.
    assert next(s for s in tile1 if s.name == "conv_compute").work[
        "vector@r0:s0"] == 256 * 8 * 17
    assert "barrier" in [s.name for s in tile1]
    assert "state_rewrite" not in [s.name for s in tile1]

    # Owner 0 owns vector units 0-1, owner 1 owns 2-3 in this geometry.
    for stage in tiles0[0] + tile1:
        if "vector@r0:s0" in stage.work:
            owner = 0 if stage in tiles0[0] else 1
            assert stage.placement["vector@r0:s0"] == (2 * owner, 2 * owner + 1)

    plan = conv.build(op, geometry(2))
    assert stage_work(program, "hbm") == pytest.approx(plan.work.resources["hbm"])
    assert stage_work(program, "vector") == pytest.approx(plan.work.resources["vector"])


def test_source_grid_y_channel_blocks_follow_cuda_x_fastest_order():
    # channels=257 -> two channel blocks: active 256 and active 1.
    op = conv.make_op(base_params(lengths=[8], channels=257, initial_state=False))
    program = inspect_program(op, compute_units=2)
    assert len(program.programs) == 2
    # x is one flat chunk, so linear CTA 0 is feature block 0 and CTA 1 is
    # feature block 1.  The 256-lane block carries more payload than the tail.
    hbm = []
    for stream in program.programs:
        stages = stream.tile(0)
        hbm.append(sum(stage.work.get("memory@r0:s0", 0) for stage in stages))
    assert hbm[0] > hbm[1] > 0
    assert hbm[0] - hbm[1] == (256 - 1) * (4 * 2 + 8 * 2 + 3 * 2 + 3 * 2 + 8 * 2)


def test_null_block_can_skip_a_later_sequence():
    op = conv.make_op(base_params(
        lengths=[3, 4], channels=130, initial_state=False, bias=False,
        has_null_block=True, cache_indices=[1, 0]))
    meta = dict(op.metadata)
    assert meta["cache_indices"] == (1, 0)
    assert meta["counts"]["early_return_programs"] == 1
    assert meta["counts"]["processed_programs"] == 1
    plan = conv.build(op, geometry(2))
    expected = explicit_source_work(
        [3, 4], 130, 4, [False, False], False, "silu", [False, True])
    assert plan.work.resources["hbm"] == pytest.approx(expected[0])
    assert plan.work.resources["vector"] == pytest.approx(expected[1])


def test_non_divisible_owner_geometry_is_reported_not_emulated():
    op = conv.make_op(base_params(lengths=[8], channels=256))
    build = conv.build(op, geometry(compute_units=32, vector_units=48))
    context = recipe_context(op, compute_units=32, vector_units=48)
    with pytest.raises(ModelingError, match="owner_geometry_incompatible"):
        build.program(context)


def test_owner_streams_replace_the_global_two_slot_window():
    op = conv.make_op(base_params(lengths=[8], channels=1024))
    program = inspect_program(op, compute_units=2)
    assert len(program.programs) == 2
    assert [stream.count for stream in program.programs] == [2, 2]
    assert [stream.slots for stream in program.programs] == [2, 2]
    # Four CTAs can be resident across two owners; the old single stream
    # exposed only two globally.
    assert sum(stream.slots for stream in program.programs) == 4

    four = inspect_program(op, compute_units=4)
    assert len(four.programs) == 4
    assert [stream.count for stream in four.programs] == [1, 1, 1, 1]
    assert [stream.slots for stream in four.programs] == [1, 1, 1, 1]


def test_multi_cta_streams_scale_until_physical_owners_are_exhausted():
    op = conv.make_op(base_params(lengths=[8], channels=1024))  # four CTAs
    durations = {}
    for units in (1, 2, 4, 8):
        result = verify_operator(hardware_case(units), op, conv.build)
        assert result["status"] == "passed"
        durations[units] = result["result"]["duration_ms"]
    assert durations[1] > durations[2] > durations[4]
    # Only four streams exist, so four and eight SMs expose the same work.
    assert durations[8] == pytest.approx(durations[4], rel=2e-3)


def test_large_token_count_is_lazy_and_never_materialized_up_front():
    op = conv.make_op(base_params(
        lengths=[1_000_000_000], channels=256, initial_state=False,
    ))
    program = inspect_program(op, compute_units=32)
    assert isinstance(program, ParallelProgram)
    # ceil(1e9 / 8) chunks x ceil(256 / 256) feature blocks, lazy per owner.
    assert sum(stream.count for stream in program.programs) == 125_000_000
    assert len(program.programs) == 32
    assert all(stream.slots <= conv.OWNER_CTA_WINDOW for stream in program.programs)
    meta = dict(op.metadata)["counts"]
    assert meta["programs"] == 125_000_000
    assert meta["padded_token_slots"] == 1_000_000_000


def test_large_batch_planning_is_linear_and_stream_bounded():
    # 10k one-token sequences: O(batch) prefix/count construction, not a
    # per-sequence recomputation or a per-CTA task list.
    lengths = [1] * 10_000
    op = conv.make_op(base_params(lengths=lengths, channels=256))
    meta = dict(op.metadata)
    assert meta["counts"]["programs"] == 10_000
    assert meta["counts"]["sequences"] == 10_000
    program = inspect_program(op, compute_units=32)
    assert len(program.programs) == 32
    assert sum(stream.count for stream in program.programs) == 10_000
    assert all(stream.slots <= conv.OWNER_CTA_WINDOW for stream in program.programs)


def test_cache_index_and_null_block_validation():
    failures = (
        ({"lengths": [3, 4], "cache_indices": [1]},
         "cache_indices_must_be_nonnegative_ints_per_sequence"),
        ({"cache_indices": [True]}, "cache_indices_must_be_nonnegative_ints_per_sequence"),
        ({"cache_indices": [-1]}, "cache_indices_must_be_nonnegative_ints_per_sequence"),
        ({"cache_indices": [7], "num_cache_lines": 7},
         "cache_index_exceeds_num_cache_lines"),
        ({"null_block_id": -1}, "null_block_id_must_be_nonnegative_int_or_none"),
        ({"null_block_id": True}, "null_block_id_must_be_nonnegative_int_or_none"),
        ({"has_null_block": "yes"}, "has_null_block_must_be_bool"),
        ({"has_null_block": False, "null_block_id": 0},
         "null_block_flag_disagrees_with_null_block_id"),
    )
    for changes, match in failures:
        with pytest.raises(ModelingError, match=match):
            conv.make_op(base_params(**changes))
    # A valid explicit cache line and a larger cache are accepted.
    op = conv.make_op(base_params(
        cache_indices=[5], num_cache_lines=6, has_null_block=False))
    assert dict(op.metadata)["cache_indices"] == (5,)
    assert dict(op.metadata)["num_cache_lines"] == 6


def test_program_dag_keeps_state_chain_only_at_chunk_zero():
    op = conv.make_op(base_params(lengths=[2, 9], channels=128, initial_state=True))
    program = inspect_program(op, compute_units=1)
    assert len(program.programs) == 1
    stream = program.programs[0]
    assert stream.count == 3  # (1 + 2) chunks x one feature block
    first = stream.tile(0)   # sequence 0, chunk 0, length 2 (< state_len)
    later_first = stream.tile(1)  # sequence 1, chunk 0, length 9
    later = stream.tile(2)   # sequence 1, chunk 1
    assert [stage.name for stage in first] == [
        "prep", "state_read", "barrier", "state_rewrite", "state_store",
        "conv_load", "conv_compute", "store"]
    assert [stage.name for stage in later_first] == [
        "prep", "state_read", "barrier", "state_store",
        "conv_load", "conv_compute", "store"]
    assert [stage.name for stage in later] == [
        "prep", "conv_load", "conv_compute", "store"]
    rewrite = next(stage for stage in first if stage.name == "state_rewrite")
    assert rewrite.work["vector@r0:s0"] == 256 * 4  # S=3 > L=2 and init
    hbm = sum(stage.work.get("memory@r0:s0", 0)
              for tile in (first, later_first, later) for stage in tile)
    vector = sum(stage.work.get("vector@r0:s0", 0)
                 for tile in (first, later_first, later) for stage in tile)
    plan = conv.build(op, geometry(1))
    assert hbm == plan.work.resources["hbm"]
    assert vector == plan.work.resources["vector"]

    # S > L with no prior state has no barrier and no tl.where (lines 281-307).
    no_init = conv.make_op(base_params(
        lengths=[2], channels=128, initial_state=False))
    stages = inspect_program(no_init, compute_units=1).programs[0].tile(0)
    assert [stage.name for stage in stages] == [
        "prep", "state_read", "state_store", "conv_load", "conv_compute", "store"]
    assert dict(no_init.metadata)["counts"]["barrier_instructions"] == 0


def test_all_zero_lengths_have_an_empty_grid_and_no_work():
    op = conv.make_op(base_params(lengths=[0, 0], channels=128))
    assert dict(op.metadata)["counts"]["programs"] == 0
    plan = conv.build(op, geometry())
    assert plan.work.resources == {"hbm": 0.0, "vector": 0.0}
    program = inspect_program(op, compute_units=4)
    assert program.programs == ()
    result = verify_operator(control_case(), op, conv.build)
    assert result["status"] == "passed"
    assert result["result"]["duration_ms"] == 0.0
    assert result["work_mismatches"] == {}


def test_zero_length_sequence_has_no_program_and_no_state_write():
    op = conv.make_op(base_params(lengths=[0, 3], channels=128, initial_state=True))
    counts = dict(op.metadata)["counts"]
    assert counts["chunks"] == 1
    assert counts["programs"] == 1
    assert counts["state_write_programs"] == 1
    assert counts["padded_token_slots"] == 8
    # No program for the zero-length sequence means no state read or store for it.
    only = conv.make_op(base_params(lengths=[3], channels=128, initial_state=True))
    assert conv.build(op, geometry()).work == conv.build(only, geometry()).work


def test_coordinator_cases_keep_split_dispatch_and_work_conservation():
    payload = json.loads(CASES.read_text())
    result = run_cases(conv, payload["cases"])
    assert result["status"] == "passed"
    assert result["scope"] == "structural_contract_only_no_accuracy_claim"
    assert {row["case"] for row in result["cases"]} == {
        f"causal_conv_prefill-{index}" for index in range(5)}
    assert {row["split"] for row in result["cases"]} == {
        "development", "heldout_geometry"}
    for row in result["cases"]:
        assert row["status"] == "passed"
        assert row["dispatch_mismatches"] == {}
        assert row["identity_invariant"]
        assert row["work_mismatches"] == {}


# ---------------------------------------------------------------------------
# Independent rolling numerical oracle.  It deliberately does not call the
# planner: the first function is a direct per-token convolution, the second
# reproduces the kernel's chunk/window/state branches.

def _inputs(rng, channels, total, width, lengths):
    x = rng.normal(0.0, 0.5, (channels, total)).astype(np.float32)
    weight = rng.normal(0.0, 0.5, (channels, width)).astype(np.float32)
    bias = rng.normal(0.0, 0.2, (channels,)).astype(np.float32)
    state = rng.normal(0.0, 0.5, (len(lengths), channels, width - 1)).astype(np.float32)
    return x, weight, bias, state


def rolling_reference(x, weight, bias, state, lengths, initial, activation,
                      skipped=None, fill=np.nan):
    """Direct causal convolution; history shifts once per token."""
    channels, _ = x.shape
    width = weight.shape[1]
    state_len = width - 1
    out = np.full_like(x, fill)
    final = state.copy()
    skipped = list(skipped or [False] * len(lengths))
    offset = 0
    for sequence, length in enumerate(lengths):
        if skipped[sequence]:
            offset += length
            continue
        history = state[sequence].copy() if initial[sequence] else np.zeros(
            (channels, state_len), dtype=np.float32)
        for token in range(length):
            acc = bias.copy() if bias is not None else np.zeros(
                (channels,), dtype=np.float32)
            for tap in range(width):
                # ``history`` is always the rolling window x[t-state_len:t];
                # the final tap is the current input token.
                value = (history[:, tap] if tap < state_len
                         else x[:, offset + token])
                acc = acc + weight[:, tap] * value
            if activation:
                acc = acc / (1.0 + np.exp(-acc))
            out[:, offset + token] = acc
            history = np.concatenate(
                (history[:, 1:], x[:, offset + token:offset + token + 1]), axis=1)
        # A zero-length sequence has no grid program, so its state is untouched.
        if length:
            final[sequence] = history
        offset += length
    return out, final


def source_tiled_convolution(x, weight, bias, state, lengths, initial, activation,
                             skipped=None, fill=np.nan):
    """Reproduce one program per (chunk, BLOCK_N block), including the tails."""
    channels, _ = x.shape
    width = weight.shape[1]
    state_len = width - 1
    out = np.full_like(x, fill)
    final = state.copy()
    skipped = list(skipped or [False] * len(lengths))
    offset = 0
    for sequence, length in enumerate(lengths):
        if skipped[sequence]:
            offset += length
            continue
        for chunk_offset in range((length + TILE_M - 1) // TILE_M):
            segment = min(TILE_M, length - chunk_offset * TILE_M)
            start = offset + chunk_offset * TILE_M
            for feature_block in range((channels + TILE_N - 1) // TILE_N):
                begin = feature_block * TILE_N
                end = min(channels, begin + TILE_N)
                active = end - begin
                first = chunk_offset == 0
                if first:
                    if initial[sequence]:
                        cols = [state[sequence][begin:end, k].copy()
                                for k in range(state_len)]
                    else:
                        cols = [np.zeros(active, dtype=np.float32)
                                for _ in range(state_len)]
                else:
                    # Lines 322-338: the preceding state_len x tokens.
                    cols = [x[begin:end, start - state_len + k].copy()
                            for k in range(state_len)]
                if first:
                    # Lines 151-307: state snapshot is read before the update.
                    if state_len <= length:
                        new_state = x[begin:end, offset + length - state_len:
                                      offset + length].copy()
                    else:
                        new_state = np.zeros((active, state_len), dtype=np.float32)
                        if initial[sequence]:
                            new_state[:, :state_len - length] = state[sequence][
                                begin:end, length:state_len]
                        new_state[:, state_len - length:] = x[
                            begin:end, offset:offset + length]
                    final[sequence, begin:end, :] = new_state
                preload = (bias[begin:end].copy() if bias is not None
                           else np.zeros(active, dtype=np.float32))
                for token in range(segment):
                    acc = preload.copy()
                    for tap in range(width):
                        value = (cols[tap] if tap < state_len
                                 else x[begin:end, start + token])
                        acc = acc + weight[begin:end, tap] * value
                    if activation:
                        acc = acc / (1.0 + np.exp(-acc))
                    out[begin:end, start + token] = acc
                    cols = cols[1:] + [x[begin:end, start + token].copy()]
        offset += length
    return out, final


@pytest.mark.parametrize("lengths,channels,width,initial,bias,activation,cache_indices", [
    ([3, 7, 8, 9, 17], 130, 4, [True, False, True, False, True], True, "silu", None),
    ([1, 2, 64], 256, 4, [False, True, True], False, None, None),
    ([9, 0], 257, 4, [True, False], True, "silu", None),
    ([9, 5], 7, 3, [True, True], True, "silu", None),
    ([9], 16, 2, [True], True, "silu", None),
    ([8, 9, 16], 257, 4, [True, False, True], True, "silu", None),
    ([3, 4], 130, 4, [True, True], True, "silu", [0, 1]),
])
def test_rolling_oracle_and_source_tiling_agree_on_values_and_final_state(
        lengths, channels, width, initial, bias, activation, cache_indices):
    rng = np.random.default_rng(20261028 + channels + width)
    x, weight, bias_array, state = _inputs(rng, channels, sum(lengths), width, lengths)
    skipped = ([index == 0 for index in cache_indices]
               if cache_indices is not None else [False] * len(lengths))
    expected_out, expected_final = rolling_reference(
        x, weight, bias_array if bias else None, state, lengths, initial,
        activation, skipped)
    actual_out, actual_final = source_tiled_convolution(
        x, weight, bias_array if bias else None, state, lengths, initial,
        activation, skipped)
    columns = np.concatenate([
        np.full(length, not skip, dtype=bool) for length, skip in zip(lengths, skipped)])
    np.testing.assert_allclose(actual_out[:, columns], expected_out[:, columns],
                               rtol=2e-5, atol=2e-6)
    if any(skipped):
        assert np.isnan(actual_out[:, ~columns]).all()
    np.testing.assert_allclose(actual_final, expected_final, rtol=2e-5, atol=2e-6)

    # The plan declares exactly the final-state writes the oracle exercised.
    changes = ({"has_null_block": True, "cache_indices": cache_indices}
               if cache_indices is not None else {})
    op = conv.make_op(base_params(
        lengths=lengths, channels=channels, width=width,
        initial_state=initial, bias=bias, activation=activation, **changes,
    ))
    counts = dict(op.metadata)["counts"]
    blocks = (channels + TILE_N - 1) // TILE_N
    active_sequences = sum(1 for length, skip in zip(lengths, skipped)
                           if length and not skip)
    assert counts["state_write_programs"] == active_sequences * blocks
    plan_work = conv.build(op, geometry()).work.resources
    assert plan_work["hbm"] > 0
    if any(not skip for skip in skipped):
        assert plan_work["vector"] > 0


def test_initial_and_final_state_paths_are_independent():
    rng = np.random.default_rng(20261028)
    lengths = [2, 9, 16]
    channels, width = 130, 4
    x, weight, bias, state_a = _inputs(rng, channels, sum(lengths), width, lengths)
    state_b = state_a + rng.normal(0.0, 0.1, state_a.shape).astype(np.float32)
    initial = [False, True, False]

    out_a, final_a = source_tiled_convolution(
        x, weight, bias, state_a, lengths, initial, "silu")
    out_b, final_b = source_tiled_convolution(
        x, weight, bias, state_b, lengths, initial, "silu")
    # Changing the initial state cannot change the final state once L >= S.
    np.testing.assert_allclose(final_a, final_b, rtol=0, atol=0)

    offset = 0
    # L=2 < S=3 and init false: one zero column then the first two x tokens.
    expected0 = np.concatenate(
        (np.zeros((channels, 1), dtype=np.float32), x[:, 0:2]), axis=1)
    np.testing.assert_allclose(final_a[0], expected0, rtol=0, atol=0)
    # L=9 and L=16: the last S real x tokens, independent of initial state.
    offset = 2
    np.testing.assert_allclose(final_a[1], x[:, offset + 9 - 3:offset + 9],
                               rtol=0, atol=0)
    offset += 9
    np.testing.assert_allclose(final_a[2], x[:, offset + 16 - 3:offset + 16],
                               rtol=0, atol=0)
    # First-chunk tokens can depend on the prior state; tokens after the first
    # chunk (chunk_offset > 0, lines 309-338) must depend only on x.
    seq1_start = 2
    np.testing.assert_allclose(out_a[:, seq1_start + 3:seq1_start + 9],
                               out_b[:, seq1_start + 3:seq1_start + 9],
                               rtol=0, atol=0)
    assert np.max(np.abs(out_a[:, seq1_start:seq1_start + 3] -
                         out_b[:, seq1_start:seq1_start + 3])) > 0

    # L=2 < S=3 and init true: keep old_state[:, 2:] then append x[:, 0:2].
    expected_shift = np.concatenate(
        (state_a[0][:, 2:3], x[:, 0:2]), axis=1)
    _, final_shift = source_tiled_convolution(
        x[:, 0:2], weight, bias, state_a[:1], [2], [True], "silu")
    np.testing.assert_allclose(final_shift[0], expected_shift, rtol=0, atol=0)


def test_numpy_oracle_rejects_wrong_chunk_boundary_or_state_tail():
    """An independent negative control for the oracle, not a planner check."""
    rng = np.random.default_rng(11)
    x, weight, bias, state = _inputs(rng, 8, 9, 4, [9])
    out, final = source_tiled_convolution(
        x, weight, bias, state, [9], [True], "silu")
    # Token 8 is in chunk 1 and must use x[5:8], not the initial state.
    history = x[:, 5:9]
    acc = bias.copy()
    for tap in range(4):
        acc = acc + weight[:, tap] * history[:, tap]
    acc = acc / (1.0 + np.exp(-acc))
    np.testing.assert_allclose(out[:, 8], acc, rtol=2e-5, atol=2e-6)
    # The final state is the last three tokens.
    np.testing.assert_allclose(final[0], x[:, 6:9], rtol=0, atol=0)
