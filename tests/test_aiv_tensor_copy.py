"""Real tensor lowering and Engine controls for a finite declared copy mapping."""
from dataclasses import replace

import pytest

from mindie_sim import operator_costs
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.program import RecipeBook, ResourceEnvelope
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.tensors import TensorLowering
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_mechanism_contract import hardware_case


def case(*, units=4, active=1, tile=128, slots=1, read_units=None, rates=None, ub=65536):
    raw = hardware_case()
    raw["machine"]["device"].update({
        "compute_units": max(1, units // 2),
        "rates": {"memory": 1e15, "vector": units * 1e9,
                  "mte2": 1000, "mte2_aic": 1000,
                  "mte2_aiv": units * 1e6, "mte3": units * 2e6, **(rates or {})},
        "resource_units": {"vector": units, "mte2": units, "mte2_aic": units,
                           "mte2_aiv": read_units or units, "mte3": units},
        "buffers": {"ub": ub}, "buffer_units": {"ub": units},
    })
    raw["runtime"]["kernels"] = {"families": {"tensor_copy": {
        "kind": "aiv_tensor_copy", "tile_elements": tile, "slots": slots, "active_units": active}}}
    return parse_case(raw)


def tensor_ops(parsed, *, shape=(16, 24), narrow=12, output_dtype="bf16", concat=False):
    views = (RequestView("r", 1, 0, False, "decode"),)
    builder = Builder("test", parsed.model.layer_plans[0], 0, views, parsed.model, "decode")
    tensors = TensorLowering(builder)
    source = TensorValue("", "bf16", TensorLayout.contiguous(shape))
    if concat:
        pieces = tuple(source.narrow(-1, start, narrow) for start in (0, narrow, 2 * narrow))
        tensors.concatenate_flat(pieces, "concat")
    else:
        value = source.narrow(-1, 0, narrow)
        tensors.materialize(value, "copy", reason="explicit_test_materialization", dtype=output_dtype)
    return views, builder.ops


def execute(parsed, views, ops):
    execution = Execution(parsed, CostModel(parsed), include_submission=False, trace_limit=10000)
    ready = execution.engine.submit("input", release=3)
    work = IterationWork(tuple(ops), IterationPlan(views), 1, 1, 1, None, "body", "copy_test")
    complete = execution.add(work, 0, 0, external_inputs={"": ready}).completion
    execution.engine.advance(horizon=2)
    assert not any(execution.engine.serviced.values())
    execution.engine.advance(until=(complete,))
    engine = execution.engine
    assert not engine._tasks and not engine._promises and not engine.allocations
    assert not any(engine.allocated.values())
    assert all(row["start_ms"] >= 3 for row in engine.trace if row["demand"])
    assert engine.serviced["mte2@r0:s0"] == engine.serviced["mte2_aic@r0:s0"] == 0
    return engine, complete.time - 3


def test_strided_ba_single_owner_copy_is_read_then_write_without_idle_core_borrowing():
    parsed = case(units=64, active=1, tile=256)
    views, ops = tensor_ops(parsed)
    engine, duration = execute(parsed, views, ops)
    # Selected [16,12] BF16 payload is 384 B; the [16,24] backing is not read.
    assert duration == pytest.approx(384 / 1000 + 384 / 2000)
    assert engine.serviced["memory@r0:s0"] == 768
    assert engine.serviced["mte2_aiv@r0:s0"] == engine.serviced["mte3@r0:s0"] == 384
    assert engine.serviced["vector@r0:s0"] == 0
    transfers = [e for e in engine.trace if e["demand"].get("mte2_aiv@r0:s0") or e["demand"].get("mte3@r0:s0")]
    assert transfers[0]["end_ms"] <= transfers[1]["start_ms"]
    assert all(p == (0,) for e in transfers for role, p in e["placement"].items())
    row = operator_costs(parsed, ops)["operators"][0]
    assert row["operator"]["mte2_bytes"] == 0  # old generic work was replaced
    assert row["operator"]["metadata"]["source_layout"]["strides"] == (24, 1)
    assert row["service_delta"] == {}


@pytest.mark.parametrize("active,slots", ((1, 1), (1, 2), (2, 1), (4, 2)))
def test_cast_tail_and_finite_slots_conserve_payload_and_use_declared_ub(active, slots):
    parsed = case(units=4, active=active, tile=32, slots=slots)
    views, ops = tensor_ops(parsed, shape=(5, 27), narrow=19, output_dtype="fp32")
    engine, _ = execute(parsed, views, ops)
    assert engine.serviced["mte2_aiv@r0:s0"] == 95 * 2
    assert engine.serviced["mte3@r0:s0"] == 95 * 4
    assert engine.serviced["memory@r0:s0"] == 95 * 6
    assert engine.serviced["vector@r0:s0"] == 95
    assert engine.peak_unit_allocated == {
        ("ub@r0:s0", owner): slots * 32 * 6 for owner in range(min(active, 3))}
    by_tile = {}
    for row in engine.trace:
        if row["demand"]:
            prefix, stage = row["task"].rsplit(":", 1)
            by_tile.setdefault(prefix, {})[stage] = row
    for stages in by_tile.values():
        assert stages["read"]["end_ms"] <= stages["cast"]["start_ms"]
        assert stages["cast"]["end_ms"] <= stages["write"]["start_ms"]


def test_two_slots_allow_read_write_overlap_with_same_owner_without_more_buffer():
    controls = []
    for slots in (1, 2):
        parsed = case(tile=128, slots=slots, rates={"mte3": 4e6})
        views, ops = tensor_ops(parsed, shape=(1, 512), narrow=512)
        controls.append(execute(parsed, views, ops))
    assert controls[0][1] == pytest.approx(8 * .256)
    assert controls[1][1] == pytest.approx(5 * .256)
    assert controls[0][0].serviced == controls[1][0].serviced
    assert controls[1][0].peak_allocated["ub@r0:s0"] == 2 * 128 * 2


def test_shared_read_pipe_cannot_borrow_an_idle_physical_pipe():
    parsed = case(units=4, active=2, tile=128, read_units=2, rates={"mte3": 4e15})
    views, ops = tensor_ops(parsed, shape=(1, 256), narrow=256)
    engine, duration = execute(parsed, views, ops)
    assert duration == pytest.approx(512 / 2000, abs=1e-8)
    reads = [row for row in engine.trace if row["demand"].get("mte2_aiv@r0:s0")]
    assert len(reads) == 2
    assert all(row["placement"]["mte2_aiv@r0:s0"] == (0,) for row in reads)


def test_concat_three_source_views_moves_each_selected_payload_once():
    parsed = case(active=2, tile=64, slots=2)
    views, ops = tensor_ops(parsed, shape=(16, 24), narrow=8, concat=True)
    engine, _ = execute(parsed, views, ops)
    assert engine.serviced["memory@r0:s0"] == 3 * 16 * 8 * 2 * 2
    assert len(dict(ops[0].metadata)["source_layouts"]) == 3
    assert engine.serviced["vector@r0:s0"] == 0


@pytest.mark.parametrize("role", ("memory", "mte2_aiv", "mte3", "vector"))
def test_declared_bandwidth_or_cast_rate_changes_time_not_physical_work(role):
    durations, services = [], []
    for factor in (.5, 1, 2):
        parsed = case(active=2, tile=32, slots=2, rates={role: 10000 * factor})
        views, ops = tensor_ops(parsed, shape=(5, 27), narrow=19, output_dtype="fp32")
        engine, duration = execute(parsed, views, ops)
        durations.append(duration)
        services.append(engine.serviced)
    assert durations[0] > durations[1] > durations[2]
    assert all(item == pytest.approx(services[0]) for item in services)


@pytest.mark.parametrize("startup,minimum", ((2, 0), (2, 10)))
def test_same_dtype_copy_has_one_outer_vector_invocation_even_with_many_tiles(startup, minimum):
    from mindie_sim.schema.invocation import InvocationTiming
    parsed = case(active=2, tile=32, slots=2)
    views, ops = tensor_ops(parsed, shape=(16, 24), narrow=12)
    plain_engine, plain_duration = execute(parsed, views, ops)
    die = replace(parsed.machine.dies[0], kernel_invocations=(("vector", InvocationTiming(
        startup_ms=startup, minimum_ms=minimum)),))
    timed = replace(parsed, machine=replace(parsed.machine, dies=(die,)))
    engine, duration = execute(timed, views, ops)
    assert duration == pytest.approx(max(minimum, plain_duration + startup))
    assert engine.serviced == plain_engine.serviced
    assert sum(e["task"].endswith(":startup") for e in engine.trace) == 1


def test_default_generic_gpu_copy_does_not_acquire_aiv_class():
    parsed = parse_case(hardware_case())
    _, ops = tensor_ops(parsed)
    book = RecipeBook.from_mapping(parsed.runtime.kernels)
    assert isinstance(book.select(ops[0]), ResourceEnvelope)
    result = operator_costs(parsed, ops)["operators"][0]
    assert result["serviced"] == {"memory@r0:s0": 768}
    assert "program_execution" not in result["operator"]["metadata"]


def test_selecting_tensor_copy_family_does_not_reclassify_other_dma():
    from mindie_sim.schema.invocation import InvocationTiming
    parsed = case()
    _, ops = tensor_ops(parsed)
    dma = replace(ops[0], family="transfer", recipe="independent_dma", metadata=())
    die = replace(parsed.machine.dies[0], kernel_invocations=(("vector", InvocationTiming(minimum_ms=10000)),))
    parsed = replace(parsed, machine=replace(parsed.machine, dies=(die,)))
    result = operator_costs(parsed, (dma,))["operators"][0]
    assert result["program"] == "ResourceEnvelope"
    assert "invocation_models" not in result["basis"]
    assert result["serviced"]["mte2@r0:s0"] == 384
    assert "mte2_aiv@r0:s0" not in result["serviced"]


def test_explicit_invocation_class_takes_priority_over_planner_class():
    from mindie_sim.schema.invocation import InvocationTiming
    parsed = case(tile=256)
    _, ops = tensor_ops(parsed)
    op = replace(ops[0], metadata=tuple({**dict(ops[0].metadata), "invocation_class": "cube"}.items()))
    die = replace(parsed.machine.dies[0], kernel_invocations=(
        ("vector", InvocationTiming(minimum_ms=10)), ("cube", InvocationTiming(minimum_ms=20))))
    parsed = replace(parsed, machine=replace(parsed.machine, dies=(die,)))
    assert operator_costs(parsed, (op,))["operators"][0]["duration_ms"] == 20
