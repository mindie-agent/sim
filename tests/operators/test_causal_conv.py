"""Causal convolution dispatch, source ownership, and history traffic.

[Role] Exercise Triton and native Ascend convolution and the governed ABI.
[Invariants] Conserve active rows, history payloads, and owner-local service.
[Timeouts & Locks] Bounded planning and local Engine execution.
[Side-Effect Policy] Read-only fixture inputs and in-memory simulation.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from _support.axes import REPRESENTATIVE_SKUS, SKUS
from _support.hardware import machine
from _support.linear_cases import case, conv_graph as graph, execute_one, gdn_decode_source_case as source_case, kda_hardware_case
from mindie_sim import Case, operator_costs
from mindie_sim.catalog.weights import stage_weight_bytes, stage_weight_terms
from mindie_sim.prediction import conv_state_copy
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.gdn_decode_programs import CONV, conv_plan, conv_task_runs, update_geometry
from mindie_sim.prediction.instructions import REGISTER_RESOURCES
from mindie_sim.prediction.mechanism import Geometry, PlannedRecipe
from mindie_sim.prediction.operators import causal_conv_prefill as conv, native_causal_conv_prefill as native
from mindie_sim.prediction.operators.contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.linear_attention import build, resolve_operator_plan
from mindie_sim.prediction.program import ParallelProgram, PipelineProgram, RecipeBook, RecipeContext
from mindie_sim.prediction.results import OpOccurrence
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.structure import export_structure, target_graphs
from mindie_sim.structure_contracts import short_conv_active_contract
from tools.run_operator_cases import control_case

pytestmark = pytest.mark.tier0

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "tests/fixtures/operator_cases/native_causal_conv_prefill.json"
TILE_M, TILE_N = 8, 256


def base_params(**changes):
    return {"lengths": [3], "channels": 128, "width": 4, "dtype": "bf16",
            "initial_state": False, "bias": False, "activation": "silu",
            "implementation": "triton_causal_conv1d_fwd", "is_apc_enabled": False,
            "has_null_block": False, **changes}


def inspect_program(op, units):
    names = {"hbm": "memory@r0:s0", "vector": "vector@r0:s0"}
    counts = {"hbm": units, "vector": 2 * units}
    ctx = RecipeContext(op, {}, names, dict.fromkeys(names, 1e12), units, resource_units=counts)
    return conv.build(op, Geometry(units, counts)).program(ctx)


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


def explicit_source_work(lengths, channels, width, initial, bias, activation, skipped):
    """Enumerate token tiles and channel blocks from the source grid."""
    history = width - 1
    np2 = 1 << (history - 1).bit_length()
    per_token = 2 + 2 * width + history + 4 * bool(activation)
    hbm = vector = barriers = 0
    for length, has_initial, null in zip(lengths, initial, skipped, strict=True):
        for row in range(0, length, TILE_M):
            tokens = min(TILE_M, length - row)
            for column in range(0, channels, TILE_N):
                hbm += 20
                if null:
                    continue
                active = min(TILE_N, channels - column)
                payload = width + bias + 2 * tokens
                if row:
                    payload += history
                else:
                    hbm += 1 + 4 * (length >= history)
                    payload += history + history * has_initial + min(length, history)
                    if length < history and has_initial:
                        payload += history - length
                        vector += TILE_N * np2
                    barriers += bool(length >= history or has_initial)
                hbm += payload * active * 2
                vector += TILE_N * (tokens * per_token + bias)
    return hbm, vector, barriers


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
    plan = conv.build(op, Geometry(32, {"vector": 64}))
    skipped = [slot == 0 for slot in cache_indices] if cache_indices else [False] * len(lengths)
    expected_hbm, expected_vector, barriers = explicit_source_work(lengths, channels, width, initial, bias, activation, skipped)
    assert plan.work.resources == {"hbm": expected_hbm, "vector": expected_vector}
    assert dict(op.metadata)["counts"]["barrier_instructions"] == barriers
    program = inspect_program(op, 4)
    assert sum(stream.count for stream in program.programs) == dict(op.metadata)["counts"]["programs"]
    stages = tuple(iter_stages(program))
    for role, amount in (("memory", expected_hbm), ("vector", expected_vector)):
        assert sum(stage.work.get(f"{role}@r0:s0", 0) for stage in stages) == amount

def test_selected_prefill_dispatch_and_default_null_block_conserve_work():
    params = base_params()
    params.pop("has_null_block")
    null = conv.make_op(params)
    assert conv.build(null, Geometry(32, {"vector": 64})).work.resources == {"hbm": 20, "vector": 0}
    assert dict(null.metadata)["counts"]["state_write_programs"] == 0
    active = conv.make_op(dict(params, cache_indices=[1]))
    report = verify_operator(control_case(units=2), active, conv.build, input_ready_ms=7)
    assert report["status"] == "passed" and report["input_dependency_obeyed"]
    assert report["buffers_released"] and report["work_mismatches"] == {}
    for change in ({"is_apc_enabled": True}, {"implementation": "triton_causal_conv1d_update"},
                   {"has_null_block": True, "null_block_id": None}, {"cache_indices": [1, 2]},
                   {"cache_indices": [-1]}, {"null_block_id": -1}, {"block_size_to_align": 7}):
        with pytest.raises(ModelingError):
            conv.make_op(dict(params, **change))


def _params(**changes):
    return {
        "lengths": [8], "dim": 16, "width": 4, "dtype": "bf16",
        "activation": "silu", "bias": False, "metadata_placement": "cpu",
        "has_initial_state": [False], "cache_indices": [1],
        "state_history_slots": 6, **changes,
    }


def _hardware(*, aic=1, aiv=1, split_reads=False, read=1e12, write=1e12, vector=1e12):
    raw = kda_hardware_case(units=aic, split_copy=split_reads)
    rates = {"memory": 1e20, "vector": vector, "mte3": write, **dict.fromkeys(REGISTER_RESOURCES, vector)}
    reads = {"mte2_aic": 1e20, "mte2_aiv": read} if split_reads else {"mte2": read}
    units = dict.fromkeys(rates.keys() | reads.keys(), aiv)
    if split_reads:
        units["mte2_aic"] = aic
    raw["machine"]["device"].update(
        compute_peaks={"bf16": 1e14, "fp16": 1e14}, rates=rates | reads, resource_units=units,
        service_modes={"reg_issue_slots": {"resource": "vector", "rate": vector}},
        buffers={"ub": 253952}, buffer_units={"ub": aiv},
    )
    return raw


def _start(params, hardware, *, release_ms=0):
    case, op = Case.from_mapping(hardware), native.make_op(params)
    execution = Execution(
        case, CostModel(case),
        recipes=RecipeBook({op.recipe: PlannedRecipe(native.build)}),
        trace_limit=2000,
    )
    work = IterationWork(
        (op,), IterationPlan((RequestView(
            "r", sum(params["lengths"]), 0, True, "prefill"),)),
        1, 1, 1, None, "prefill", "source-control",
    )
    return execution, execution.add(work, 0, release_ms)


def _finish(execution, step):
    execution.engine.advance(until=(step.completion,))
    assert execution.engine.summary()["live_tasks"] == 0
    assert not any(execution.engine.summary()["live_allocated"].values())
    return execution.engine.trace


def _stage(rows, suffix):
    return next(row for row in rows if row["task"].endswith(suffix))


@pytest.mark.parametrize("metadata", ["cpu", "device"])
@pytest.mark.parametrize("bias", [False, True])
def test_token_prefetch_respects_outer_ready_and_history(metadata, bias):
    execution, step = _start(
        _params(metadata_placement=metadata, bias=bias, has_initial_state=[True]),
        _hardware(read=1000, write=1000, vector=1e20), release_ms=7,
    )
    execution.engine.advance(horizon=6.5)
    assert not any(execution.engine.summary()["serviced"].values())
    rows = _finish(execution, step)
    gate = _stage(rows, ":pairs:program:begin")
    load = _stage(rows, ":pairs:program:tile1:load")
    history = _stage(rows, ":history")
    assert load["start_ms"] == pytest.approx(gate["end_ms"])
    assert gate["end_ms"] >= history["end_ms"] >= 7


def test_prefetch_holds_source_ub_buffers_until_kernel_drain():
    execution, step = _start(
        _params(lengths=[1], metadata_placement="device", has_initial_state=[True]),
        _hardware(read=1000, write=1000, vector=1e20),
    )
    execution.engine.advance(horizon=1)
    during = execution.engine.summary()
    resident = (5 + 2) * 4096 * 2 + 8 * 4096 * 4
    assert during["serviced"]["mte2@r0:s0"] == pytest.approx(1)
    assert during["live_allocated"]["ub@r0:s0"] == resident
    assert during["peak_allocated_per_unit"]["ub@r0:s0:unit0"] == resident
    rows = _finish(execution, step)
    reserve = _stage(rows, ":reserve0")
    prefetch = [row for row in rows if ":prefetch0:program:" in row["task"] and row["demand"]]
    release = _stage(rows, ":release0")
    drain_end = max(row["end_ms"] for row in rows if ":body0:" in row["task"])
    assert reserve["end_ms"] <= min(row["start_ms"] for row in prefetch)
    assert release["start_ms"] >= drain_end


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("activation", ["none", "silu"])
@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
def test_regbase_ledger_matches_selected_source_instruction_counts(width, activation, dtype):
    instructions = {
        "token_load": 1, "token_cast": 1, "weight_load": width,
        "state_load": width - 1, "multiply": width, "add": width - 1,
        "state_store": width - 1, "output_cast": 1, "output_store": 1,
        "silu": 4 if activation == "silu" else 0,
    }
    params = _params(width=width, activation=activation, dtype=dtype)
    op = native.make_op(params)
    geometry = Geometry(1, {"vector": 1}, {}, {})
    ledger = native.source_ledger(op, geometry)
    setup = (5 + 3) * 16
    issue = {2: 7, 3: 15, 4: 25}[width] + 8 * (sum(instructions.values()) + 1 + width)
    assert ledger["work"]["vector"] == setup
    assert ledger["work"]["reg_issue_slots"] == issue
    assert ledger["work"]["reg_ub_accesses"] == 3 * (width - 1) + 8 * 3 * width
    execution, step = _start(params, _hardware())
    _finish(execution, step)
    assert execution.engine.summary()["serviced"]["vector@r0:s0"] == pytest.approx(setup + issue)


@pytest.mark.parametrize("dim,port_calls", [(16, 21), (64, 21), (80, 42)])
@pytest.mark.parametrize("aiv", [1, 4])
def test_rolling_ub_ports_preserve_register_tails_and_cannot_borrow_inactive_owners(dim, port_calls, aiv):
    hardware = _hardware(aiv=aiv, read=1e20, write=1e20, vector=1e20)
    hardware["machine"]["device"]["rates"]["reg_ub_accesses"] = 1000 * aiv
    params = _params(lengths=[1], dim=dim, activation="none")
    execution, step = _start(params, hardware)
    _finish(execution, step)
    assert step.completion.time == pytest.approx(port_calls)
    assert execution.engine.serviced["reg_ub_accesses@r0:s0"] == port_calls
    hardware["machine"]["device"]["rates"]["reg_ub_accesses"] *= 2
    faster, faster_step = _start(params, hardware)
    _finish(faster, faster_step)
    assert faster_step.completion.time == pytest.approx(port_calls / 2)


@pytest.mark.parametrize("aiv", [1, 2, 8])
def test_rolling_issue_mode_owns_its_bound_physical_pipe(aiv):
    hardware = _hardware(aiv=aiv, read=1e20, write=1e20, vector=1e20)
    device = hardware["machine"]["device"]
    device["rates"]["vector_issue"] = 1000 * aiv
    device["resource_units"]["vector_issue"] = aiv
    device["service_modes"]["reg_issue_slots"] = {
        "resource": "vector_issue", "rate": 1000 * aiv,
    }
    execution, step = _start(
        _params(lengths=[1], dim=80, state_rows=0, state_history_slots=0), hardware)
    rows = _finish(execution, step)
    assert (step.completion.time, execution.engine.serviced["vector_issue@r0:s0"]) == pytest.approx((110, 110))
    for row in rows:
        if row["demand"].get("vector_issue@r0:s0"):
            assert row["placement"]["vector_issue@r0:s0"] == (0,)


@pytest.mark.parametrize("override,match", (
    ({"run_mode": 1}, "runMode=0"), ({"head_num": 1}, "head-major"),
    ({"input_mode": "batch"}, "varlen"), ({"num_accepted_tokens": [1]}, "numAcceptedTokens"),
    ({"width": 5}, "width"),
))
def test_native_rejects_unsupported_source_branches(override, match):
    with pytest.raises(ValueError, match=match):
        native.make_op(_params(**override))


def test_native_host_tiling_matches_retained_source_oracles():
    for sample in json.loads(CASES.read_text())["cases"]:
        if expected := sample.get("expected_planner"):
            actual = native.host_tiling(native.make_op(sample["params"]))
            assert {key: actual[key] for key in expected} == expected

@pytest.mark.parametrize("width", (2, 3, 4))
@pytest.mark.parametrize("bias", (False, True))
def test_short_history_tail_drains_after_actual_input_and_snapshot_load(width, bias):
    params = _params(lengths=[1], width=width, bias=bias, metadata_placement="device",
                     has_initial_state=[True], dtype="fp16")
    op = native.make_op(params)
    ledger = native.source_ledger(op, Geometry(1, {"vector": 1}, {}))
    reference = native.source_ledger(native.make_op(dict(params, dtype="bf16")), Geometry(1, {"vector": 1}, {}))
    assert ledger["work"] == reference["work"]
    execution, step = _start(params, _hardware(split_reads=True))
    rows = _finish(execution, step)
    store = _stage(rows, ":state_store")
    assert store["demand"]["mte3@r0:s0"] == (width - 1) * 16 * 2
    if bias:
        assert not any(r["task"].endswith(":state_load") for r in rows)
        assert ledger["work"].get("reg_issue_slots", 0) == 0
    else:
        load = _stage(rows, ":state_load")
        assert load["demand"]["mte2_aiv@r0:s0"] == (width - 1) * 16 * 2
        assert store["start_ms"] >= load["end_ms"]
    tile = ledger["host_tiling"]["token_block_size"]
    signal = (np.arange(tile + width + 1) % 7 / 7).astype(np.float16).astype(np.float32)
    taps = np.linspace(0.1, 0.9, width).astype(np.float16).astype(np.float32)
    def silu(values):
        return values / (1 + np.exp(-values))

    reference = silu(np.correlate(signal, taps, mode="valid") + bias)
    actual = np.concatenate([
        silu(np.lib.stride_tricks.sliding_window_view(signal[start:min(len(reference), start + tile) + width - 1], width)
             @ taps + bias).astype(np.float16)
        for start in range(0, len(reference), tile)
    ])
    np.testing.assert_array_equal(actual, reference.astype(np.float16))


def run(lengths, dim, *, compact, width, bias, speculative, limited):
    shape = (sum(lengths), dim)
    values = dict(sequence_tokens=lengths, kernel_width=width, bias=bias,
                  speculative=speculative, state_history_slots=2 + max(lengths))
    op = OpSpec("conv", "vector", "decode", 0, 0, (), shape, shape, shape, shape,
                dtype="bf16", metadata=tuple(values.items()))
    rates = {"hbm": 1e8, "mte2_aiv": 1e8, "mte2_aic": 1e8,
             "vector": 1e8, "mte3": 1e8, limited: 1e6}
    counts = dict.fromkeys(rates, 1)
    geom = Geometry(1, counts, {"ub": 256 * 1024}, {"ub": 1})
    plan = conv_plan(op, geom, compact=compact)
    engine = Engine([Resource(role, rate) for role, rate in rates.items()], pools=geom.buffers)
    ctx = RecipeContext(plan.work.apply(op), {}, {role: role for role in rates}, rates, 1,
                        resource_units=counts, pool_names={"ub": "ub"}, pool_units=geom.buffer_units)
    ready = engine.submit("input", release=3)
    done = plan.lower(ctx).start(engine, "conv", ready)
    engine.advance(horizon=2)
    assert not any(engine.serviced.values())
    engine.advance(until=(done,))
    assert not engine._tasks and not engine._promises and not engine.allocations
    assert engine.serviced["mte2_aic"] == 0
    for role, amount in plan.work.resources.items():
        assert engine.serviced[role] == pytest.approx(amount)
    return engine, done.time - 3


@pytest.mark.parametrize("lengths,dim,units", (
    ((4,) * 128, 2560, 64), ((4,) * 40, 1280, 64),
    ((4, 3, 2) * 10, 1297, 4), ((4, 4, 3, 3, 2, 2), 513, 8),
))
def test_runs_preserve_source_round_robin_owners_and_tail_lengths(lengths, dim, units):
    tile, columns, blocks = update_geometry(len(lengths), dim, units)
    for owner in range(blocks):
        expected = [(lengths[index // columns], min(tile, dim - (index % columns) * tile))
                    for index in range(owner, len(lengths) * columns, blocks)]
        actual = [(q, n) for q, n, repeat in conv_task_runs(
            lengths, dim, tile, columns, blocks, owner) for _ in range(repeat)]
        assert actual == expected


@pytest.mark.parametrize("width,speculative,bias", ((2, False, False), (3, False, True), (4, True, True)))
@pytest.mark.parametrize("limited", ("mte2_aiv", "mte3", "vector"))
def test_one_owner_serial_repetition_preserves_duration_and_service(width, speculative, bias, limited):
    args = dict(width=width, speculative=speculative, bias=bias, limited=limited)
    expanded, original = run((4,) * 12, 512, compact=False, **args)
    grouped, duration = run((4,) * 12, 512, compact=True, **args)
    assert (duration, grouped.serviced) == (pytest.approx(original, rel=1e-10), pytest.approx(expanded.serviced))
    assert grouped.completed_count < expanded.completed_count / 5

@pytest.mark.parametrize("batch", (1, 4))
def test_update_binds_active_source_rows_and_resident_weights(batch):
    raw, baseline = case(batch), case(batch, native=False)
    parsed, plain = parse_case(raw), parse_case(baseline)
    ops = graph(raw, batch)
    convs = [op for op in ops if op.recipe == CONV]
    layers = {layer.index for layer in parsed.model.layer_plans if layer.attention == "kda"}
    assert {op.layer for op in convs} == layers and len(convs) == len(layers)
    for op in convs:
        meta = dict(op.metadata)
        assert op.local_shape == (batch, 4608) and op.source_boundary == "kernel"
        assert meta["sequence_tokens"] == (1,) * batch and meta["state_write_bytes"] == batch * 3 * 4608 * 2
    row = operator_costs(raw, convs[:1], input_ready_ms=9)["operators"][0]
    assert row["operator"]["hbm_bytes"] == batch * 8 * 4608 * 2
    assert row["operator"]["mte3_bytes"] == batch * 4 * 4608 * 2
    assert not row["service_before_input"] and row["live_tasks"] == 0
    prepared = [term for group in stage_weight_terms(parsed.model, parsed.runtime.parallel)
                for term in group if "kda_prepared_conv" in term.name]
    assert len(prepared) == len(layers) and all(not term.source_owned for term in prepared)
    assert (sum(stage_weight_bytes(parsed.model, parsed.runtime.parallel))
            - sum(stage_weight_bytes(plain.model, plain.runtime.parallel))) == len(layers) * 4 * 4608 * 2


def physical_case():
    raw = kda_hardware_case(units=2, split_copy=True)
    raw["machine"]["device"].update({
        "rates": {"memory": 1e9, "vector": 1e9, "mte2": 1e9,
                  "mte2_aic": 1e9, "mte2_aiv": 4e6, "mte3": 8e6},
        "resource_units": {"vector": 4, "mte2_aic": 4, "mte2_aiv": 4, "mte3": 4},
        "buffers": {"ub": 4096}, "buffer_units": {"ub": 4}})
    raw["runtime"]["kernels"] = {"lowerings": {"conv_state_copy": {
        "implementation": conv_state_copy.IMPLEMENTATION, "options": {"state_slots": 3}}}}
    return parse_case(raw)


def copy_op(direction, active):
    shape = (len(active), 3, 257)
    metadata = dict(direction=direction, active_requests=active, cache_index_dtype="int32",
                    query_start_dtype="int64", invocation_class="vector")
    return OpSpec("copy", "conv_state_copy", "decode", 0, 0, (), shape, shape, shape, shape,
                  dtype="bf16", recipe=conv_state_copy.IMPLEMENTATION, source_boundary="kernel",
                  deps=("input",), metadata=tuple(metadata.items()))


@pytest.mark.parametrize("direction", ("gather", "scatter"))
@pytest.mark.parametrize("active", ((True, True), (False, True), (False, False)))
def test_copy_tail_and_inactive_masks_match_source_memory_direction(direction, active):
    parsed, op = physical_case(), copy_op(direction, active)
    plan = conv_state_copy.build(op, Geometry.from_devices(parsed.machine.dies))
    row = operator_costs(parsed, (op,), input_ready_ms=7)["operators"][0]
    reads = (2 if direction == "scatter" else sum(active)) * 771 * 2
    writes = (sum(active) if direction == "scatter" else 2) * 771 * 2
    packed_indices = 8 if direction == "gather" else 0
    assert plan.work.resources["hbm"] == reads + writes + packed_indices + 8 * 20
    assert (plan.work.resources["mte2_aiv"], plan.work.resources["mte3"]) == (reads, writes + packed_indices)
    assert not row["service_before_input"] and row["live_tasks"] == 0
    assert row["service_delta"] == pytest.approx(dict.fromkeys(row["service_delta"], 0), abs=1e-8)


@pytest.mark.parametrize("keys,heads,tokens", ((2, 6, 16), (4, 12, 128)))
def test_projection_materialization_copies_only_the_strided_gate_payload(keys, heads, tokens):
    parsed, views, ops = source_case((4,) * (tokens // 4), keys=keys, heads=heads)
    copy = next(op for op in ops if op.recipe == "z_norm_materialize")
    meta = dict(copy.metadata)
    width, value = (2 * keys + heads) * 128, heads * 128
    assert meta["source_layout"]["strides"] == (width + value, 1)
    assert meta["source_layout"]["offset"] == width
    assert copy.mte2_bytes == copy.mte3_bytes == tokens * value * 2
    norm = next(op for op in ops if op.recipe == "linear_output_norm_gate")
    assert copy.op_id in norm.deps
    engine, _ = execute_one(parsed, views, copy)
    assert engine.serviced["memory@r0:s0"] == pytest.approx(tokens * value * 4)


@pytest.mark.parametrize("lengths,heads", (((1,), 12), ((4,), 1)))
def test_singleton_compatible_z_reshape_is_a_view(lengths, heads):
    _, _, ops = source_case(lengths=lengths, keys=1, heads=heads)
    assert not any(op.recipe == "z_norm_materialize" for op in ops)
    norm = next(op for op in ops if op.recipe == "linear_output_norm_gate")
    projection = next(op for op in ops if op.recipe == "linear_qkvz")
    assert projection.op_id in norm.deps


def test_source_z_reshape_uses_padded_projection_rows_without_padding_recurrence():
    parsed, views, _ = source_case(lengths=(1,), keys=4, heads=12)
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, mode="decode", tokens=16)
    copy = next(op for op in ops if op.recipe == "z_norm_materialize")
    assert copy.hbm_bytes == 16 * 1536 * 4
    recurrence = next(op for op in ops if op.family == "recurrent_vector")
    assert dict(recurrence.metadata)["tokens"] == 1
    core_copy = next(op for op in ops if op.recipe == "core_output_copy")
    assert core_copy.hbm_bytes == 1536 * 4


@pytest.mark.parametrize("model,tp,dim", (("qwen36", 8, 1280), ("qwen38", 4, 2560)))
def test_source_convolution_exports_active_boundaries_and_layer_shard(model, tp, dim):
    path = ROOT / f"examples/cases/high-concurrency-20261030/source-gdn/{model}-a5-tp{tp}-c4.json"
    raw = json.loads(path.read_text())
    nodes = [node for node in export_structure(raw, phase="decode", tokens=4, context=4104,
                                               sequences=4, scope="body")["nodes"]
             if node["family"] == "short_convolution"]
    expected = ["causal_conv1d_active", "CausalConv1d", [[16, dim], "bf16", "ND"],
                [[4, dim], "bf16", "ND"], [[], None, None],
                [[6, dim], "bf16", "ND"], [[16, dim], "bf16", "ND"]]
    assert len(nodes) == 48 and all(row["contract"] == expected for row in nodes)
    op = nodes[0]["operator"]
    changed = deepcopy(op)
    changed["metadata"]["state_history_slots"] = 7
    assert short_conv_active_contract(changed) != short_conv_active_contract(op)


@pytest.fixture(params=SKUS)
def planning(request):
    sku = request.param
    geom = Geometry.from_devices(machine(sku).dies)
    return sku, geom, OperatorPlanningContext(OpOccurrence(0, 0, 0, "conv"), geom, geom.datapath)


def _governed_op(**metadata):
    values = {"lengths": (1, 4), "channels": 16, "width": 4,
              "has_bias": True, "activation": "silu", "speculative_width": 4, **metadata}
    return OpSpec("conv", "causal_conv1d", "decode", 0, 0, (), (), (), (), (),
                  recipe="causal_conv1d", dtype="bf16", metadata=tuple(values.items()))


@pytest.mark.parametrize("mode,extra", (("varlen", 0), ("incremental", 3)))
def test_governed_conv_conserves_separate_phases_and_speculative_history(planning, mode, extra):
    _, geom, ctx = planning
    op = _governed_op(mode=mode)
    plan = resolve_operator_plan(op, ctx).recipe
    assert plan.work == build(op, geom).work
    tokens, channels, width, item = 5, 16, 4, 2
    history = 2 * (width - 1) * channels * item
    payload = tokens * channels * item
    expected = {
        "input": ("hbm", payload), "history": ("hbm", history),
        "weights": ("hbm", channels * width * item),
        "convolve": ("vector", 2 * channels * tokens * width),
        "bias": ("hbm", channels * item), "add_bias": ("vector", tokens * channels),
        "silu": ("vector", 4 * tokens * channels), "output": ("hbm", payload),
        "state_writeback": ("hbm", history + extra * channels * item),
    }
    phases = {p.phase_id: p for p in plan.pipeline_spec.phases}
    assert phases.keys() == expected.keys()
    for name, (pipe, amount) in expected.items():
        assert sum(w.amount for w in phases[name].work if w.resource == pipe) == amount
    assert set(phases["convolve"].after) == {"input", "history", "weights"}
    assert set(phases["state_writeback"].after) == {"history", "input", "output"}


def test_governed_update_selects_isa_and_checks_tensor_shapes(planning):
    sku, _, ctx = planning
    for invalid in ({"weight_shape": (4, 16)}, {"history_shape": (4, 16)}):
        with pytest.raises(ModelingError, match="weight_or_history_shape_mismatch"):
            resolve_operator_plan(_governed_op(**invalid), ctx)
    op = _governed_op(abi="causal_conv1d_update", weight_shape=(16, 4), history_shape=(2, 3, 16))
    if sku == REPRESENTATIVE_SKUS["a5"]:
        result = resolve_operator_plan(op, ctx)
        assert result.implementation == "causal_conv1d_update"
        assert result.recipe.metadata["mode"] == "incremental"
        assert result.recipe.metadata["history_rows"] == 3
    else:
        with pytest.raises(ModelingError, match="requires Arch35"):
            resolve_operator_plan(op, ctx)


def test_strided_state_copies_bracket_the_original_convolution():
    path = ROOT / "examples/cases/high-concurrency-20261030/glm-flash-ws9/a5_glm53flash_tp8_c16_p4k.json"
    raw = json.loads(path.read_text())
    raw["runtime"]["kernels"].setdefault("lowerings", {})["conv_state_copy"] = {
        "implementation": conv_state_copy.IMPLEMENTATION, "options": {"state_slots": 6}}
    ops = target_graphs(raw, scope="body", phase="decode", tokens=4, context=4096, sequences=16)[0]
    copies = [op for op in ops if op.recipe == conv_state_copy.IMPLEMENTATION]
    convs = {op.layer: op for op in ops if op.recipe == "short_convolution"}
    assert len(copies) == 2 * len(convs) == 68
    for op in copies:
        meta = dict(op.metadata)
        assert op.local_shape == (16, 6, 3072) and meta["state_strides"] == (1, 6)
        conv = convs[op.layer]
        assert (op.op_id in conv.deps if meta["direction"] == "gather" else op.deps == (conv.op_id,))
