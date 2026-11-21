"""Kernel invocation boundaries, instruction services, and host submission.

[Role]
  Pin the causal laws around one device invocation: declared startup and
  completion floors, execution-class selection, source-call expansion,
  instruction capacity and owner-local PIPE_S/scalar services, and the host
  enqueue/replay policy that stays outside per-kernel timing.
[Invariants]
  * Startup is a delay after original readiness; the minimum is concurrent with
    startup+body, so duration is ``max(minimum, startup + body)``.
  * Logical joins, pure DMA and host submission never acquire an AIV kernel
    floor; mixed classes pay one maximum, never two stacked dispatches.
  * Instruction capacity counts issued calls on their declared owner; a typed
    service is owner-local and undeclared scalar work stays visible but unpriced.
  * Every expected time is hand-derived below from declared rates and units.
[Timeouts & Locks] Pure in-memory Engine runs; no threads, sleeps or locks.
[Side-Effect Policy] No filesystem, environment or network writes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, operator_costs, predict, target_graphs
from mindie_sim.prediction.call import call_costs
from mindie_sim.prediction.cost import CostModel, primitive_key
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.execution import Execution, SubmissionPolicy
from mindie_sim.prediction.operators.grouped_moe.grouped_dequant import build as build_grouped_dequant
from mindie_sim.prediction.instructions import (
    CONTROL_INSTRUCTIONS, InstructionClass, InstructionCount, InstructionInventory,
    PrimitiveInventory)
from mindie_sim.prediction.invocation import InvocationProgram
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.primitives import PrimitiveStore
from mindie_sim.prediction.program import KernelProgram, RecipeBook, RecipeContext, Stage
from mindie_sim.prediction.services import ResourceService
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.invocation import InvocationTiming
from mindie_sim.schema.operators import IterationWork, OpSpec
from mindie_sim.schema.workload import IterationPlan, RequestView

from _support.collective_cases import cohort_case, exchange
from _support.hardware import hardware_case, sku_case
from _support.kernel_invocation import copy_op, curve, declare, op
from _support.memory_cases import case as serving_case
from _support.memory_cases import linear_work

pytestmark = pytest.mark.tier0

SERVICE_SKU = "Ascend950DT_9582"
DRAM_SERVICE_SCALE = 1 / .8316  # A5 sustainable DRAM service vs raw HBM peak
GROUPED_ROWS = (8,) * 15 + (0,)

class RecordingBody:
    """Body fake recording engine time/owner; ``phases`` expands independent work."""
    def __init__(self, work=None, phases=0):
        self.work, self.phases, self.calls = work or {}, phases, []

    def start(self, engine, name, ready, *, owner=""):
        self.calls.append((engine.now, ready.done, owner))
        if not self.phases:
            return engine.submit(name + ":service", self.work, after=(ready,), owner=owner)
        events = tuple(engine.submit(f"{name}:phase{i}", {"cube": 1}, after=(ready,),
                                     owner=owner) for i in range(self.phases))
        return engine.submit(name + ":complete", after=events, owner=owner)

def timed_case():
    raw = hardware_case()
    raw["machine"]["device"]["kernel_invocations"] = {
        "vector": {"minimum_ms": .0025}, "cube": {"minimum_ms": .005}}
    return raw

def vector(name="v", *, amount=1, **kwargs):
    return replace(op(name, **kwargs), vector_ops=amount)

def copy_lower(operation):
    def lower(plan, model, runtime, machine):
        return IterationWork((operation,), plan, 1, 1, 1, None, "prefill",
                             "invocation_control")
    return lower

def call_recipe_book():
    def children(parent):
        return (replace(op("a", memory=1e6), source_boundary="kernel", recipe="inner",
                        vector_ops=1),
                replace(op("b", memory=2e6, deps=("a",)), source_boundary="kernel",
                        recipe="inner", flops=1))

    def prepare(operation, geometry):
        work = Work(resources={"hbm": operation.hbm_bytes, "vector": operation.vector_ops},
                    matrix={"bf16": operation.flops})
        return OperatorPlan(work, lambda context: KernelProgram(
            (Stage("work", context.bind(work)),)))
    return RecipeBook({"outer": PlannedRecipe(prepare, kernels=children),
                       "inner": PlannedRecipe(prepare)})

def call_op():
    return replace(op("closure"), source_boundary="call", recipe="outer")

def grouped_case(**intervals):
    raw = sku_case(SERVICE_SKU, device={"resource_units": {"vector": 9}}).to_dict()
    for device in raw["machine"]["devices"]:
        arch = dict(device.get("compute_arch") or {})
        arch["instruction_services"] = {**dict(arch.get("instruction_services") or {}), **{
            role: {"unit": "calls", "owner_axis": "vector", "issue_ns": ns,
                   "provenance": {"kind": "synthetic_capacity_control"}}
            for role, ns in intervals.items()}}
        device["compute_arch"] = arch
    return parse_case(raw)

def grouped_row(**intervals):
    shape = (sum(GROUPED_ROWS), 512)
    operation = OpSpec(
        "dequant", "grouped_quant", "decode", 0, 0, ("r",), shape, shape, shape, shape,
        recipe="grouped", metadata=(("grouped_dequant", {
            "expert_rows": GROUPED_ROWS, "width": 512, "input_capacity_rows": 2048}),))
    return operator_costs(grouped_case(**intervals), (operation,),
                          recipes=RecipeBook({"grouped": PlannedRecipe(
                              build_grouped_dequant)}))["operators"][0]

def capacity_execute(inventory, classes, *, vector_rate=100, owners=(0,)):
    engine = Engine([Resource("vector", vector_rate, units=2), Resource("writes", 4, units=2),
                     Resource("ub", 4, units=2), Resource("stores", 2, units=2)])
    roles = {"vector": "vector", "reg_data_writes": "writes",
             "reg_ub_accesses": "ub", "reg_ub_stores": "stores"}
    context = RecipeContext(None, {}, roles,
                            {name: value.rate for name, value in engine.resources.items()}, 1,
                            resource_units={role: 2 for role in roles},
                            service_modes={"reg_issue_slots": ("vector", 4000)})
    work = context.bind(inventory.capacity_work(classes))
    end = KernelProgram(tuple(
        Stage(str(owner), work, placement=context.owned(owner, 2, tuple(roles)))
        for owner in owners)).start(engine, "registers", engine.submit("ready"))
    engine.advance(until=(end,))
    return end.time, engine.serviced

def service_hardware(scale=1, *, linked=False):
    services = {
        "mte2_descriptor_issue": {"unit": "descriptors", "owner_axis": "vector",
            "issue_ns": 1000 / scale,
            **({"scale_with": {"resource": "mte2_aiv",
                               "reference_per_owner_rate": 100e9}} if linked else {}),
            "provenance": {"kind": "test_control"}},
        "vbitsort32": {"unit": "blocks32", "owner_axis": "vector",
                       "per_owner_rate": 1e6 * scale}}
    rates = {"hbm": 1000, "vector": 1000, **({"mte2_aiv": 4 * 100e9} if linked else {})}
    units = {"vector": 4, **({"mte2_aiv": 4} if linked else {})}
    device = {"memory_bytes": 1000, "compute_peaks": {"bf16": 1000}, "rates": rates,
              "compute_units": 2, "resource_units": units,
              "compute_arch": {"instruction_services": services}}
    return Hardware.from_mapping({"num_devices": 1, "device": device})

def service_execute(scale, owners):
    device = service_hardware(scale).dies[0]
    roles = ("mte2_descriptor_issue", "vbitsort32")
    rates = {role: dict(device.rates)[role] / 1000 for role in roles}
    units = {role: device.units(role) for role in roles}
    engine = Engine([Resource(role, rates[role], units=units[role]) for role in roles])
    context = RecipeContext(None, {}, {role: role for role in roles}, rates, 2,
                            resource_units=units)
    demand = context.bind(InstructionInventory({
        "DataCopyRead": InstructionCount(calls=8),
        "VBitSort32": InstructionCount(calls=4)}).capacity_work(CONTROL_INSTRUCTIONS))
    ready = engine.submit("late_input", latency=7)
    done = KernelProgram(tuple(Stage(str(i), demand, placement=context.owned(owner, 4, roles))
        for i, owner in enumerate(owners))).start(engine, "calls", ready)
    engine.advance(until=(done,))
    return done.time - ready.time, engine.serviced

def linked_execution(machine):
    device = machine.dies[0]
    role = "mte2_descriptor_issue"
    engine = Engine([Resource(role, dict(device.rates)[role] / 1000,
                              units=device.units(role))])
    ready = engine.submit("input", latency=7)
    event = engine.submit("descriptors", {role: 8}, after=(ready,), placement={role: (0,)})
    engine.advance(until=(event,))
    return event.time - ready.time, engine.serviced

def pipeline_case():
    raw = serving_case(prompt=3, output=3)
    raw["pools"] = {pool: {"machine": deepcopy(raw["machine"]),
                           "runtime": deepcopy(raw["runtime"])}
                    for pool in ("prefill", "decode")}
    raw["connector"] = {"bytes_per_s": 128000, "latency_us": 0}
    return raw

def test_startup_is_a_causal_offset_after_original_readiness():
    assert InvocationTiming() == InvocationTiming(0.0, 0.0, {})
    source = {"source": "capture", "grouped_residual_ms": 40}
    declared = InvocationTiming.from_mapping({"provenance": source})
    assert declared.startup_ms == declared.minimum_ms == 0
    assert declared.to_dict() == {"startup_ms": 0.0, "minimum_ms": 0.0, "provenance": source}
    source["source"] = "mutated"
    assert declared.provenance["source"] == "capture"
    assert InvocationTiming.from_mapping({"startup_ms": 3, "minimum_ms": 5}).to_dict() == {
        "startup_ms": 3.0, "minimum_ms": 5.0, "provenance": {}}
    engine = Engine([Resource("cube", 10)], trace_limit=100)
    ready = engine.submit("input_ready", release=10)
    body = RecordingBody({"cube": 20})
    done = InvocationProgram(body, InvocationTiming(startup_ms=5)).start(
        engine, "inv", ready, owner="op")
    engine.advance(until=(ready,))
    assert engine.now == 10 and body.calls == []
    engine.advance(horizon=14.5)
    assert body.calls == []
    engine.advance(until=(done,))
    assert body.calls == [(15.0, True, "op")]
    service = next(row for row in engine.trace if row["task"] == "inv:body:service")
    assert (service["start_ms"], service["end_ms"], done.time) == (15, 17, 17)
    assert engine.serviced["cube"] == 20 and engine.busy_ms["cube"] == 2
    plain, body = Engine([Resource("cube", 10)], trace_limit=100), RecordingBody({"cube": 20})
    done = InvocationProgram(body, InvocationTiming()).start(
        plain, "plain", plain.submit("ready", release=7))
    plain.advance(until=(done,))
    assert body.calls == [(7.0, True, "")] and done.time == 9

@pytest.mark.parametrize("work,startup,minimum,expected", (
    ({"cube": 40}, 5, 7, 9), ({"cube": 40}, 5, 20, 20),
    ({"cube": 40}, 0, 4, 4), ({"cube": 40}, 2, 0, 6),
    ({}, 0, 0, 0), ({}, 3, 5, 5)))
def test_minimum_runs_concurrently_with_startup_and_body(work, startup, minimum, expected):
    engine = Engine([Resource("cube", 10)])
    done = InvocationProgram(RecordingBody(work), InvocationTiming(startup, minimum)).start(
        engine, "inv", engine.submit("ready"))
    engine.advance(until=(done,))
    assert all((done.time == pytest.approx(expected), engine.serviced["cube"] == work.get("cube", 0)))
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0

def test_independent_invocations_do_not_gain_a_global_barrier():
    engine = Engine([Resource("left", 10), Resource("right", 10)])
    ready = engine.submit("ready")
    left, right = RecordingBody({"left": 10}), RecordingBody({"right": 10})
    first = InvocationProgram(left, InvocationTiming(startup_ms=5)).start(
        engine, "first", ready, owner="a")
    second = InvocationProgram(right, InvocationTiming(startup_ms=5)).start(
        engine, "second", ready, owner="b")
    engine.advance(until=(first, second), first=False)
    assert all((first.time == second.time == 6, left.calls == [(5.0, True, "a")] and right.calls == [(5.0, True, "b")]))
    assert engine.serviced == {"left": 10, "right": 10}

def test_wrapper_preserves_work_allocation_and_completion_floor():
    def run(wrapped):
        engine = Engine([Resource("hbm", 20), Resource("cube", 10)], pools={"L1": 64},
                        pool_units={"L1": 1}, trace_limit=100)
        body = KernelProgram((
            Stage("load", {"hbm": 40}, acquire=(Allocation("buf", "L1", 64, (0,)),)),
            Stage("compute", {"cube": 40}, after=("load",),
                  free=(Allocation("buf", "L1", 64, (0,)),))))
        done = (InvocationProgram(body, InvocationTiming(startup_ms=3, minimum_ms=6)).start(
            engine, "inv", engine.submit("ready")) if wrapped
            else body.start(engine, "inv", engine.submit("ready")))
        engine.advance(until=(done,))
        return engine, done.time
    (plain, plain_ms), (wrapped, wrapped_ms) = run(False), run(True)
    assert all((plain_ms == 6 and wrapped_ms == 9, wrapped.serviced == plain.serviced == {"hbm": 40, "cube": 40}))
    assert wrapped.busy_ms == plain.busy_ms == {"hbm": 2, "cube": 4}
    assert (wrapped.peak_allocated, wrapped.allocated) == ({"L1": 64}, {"L1": 0})
    floor_engine = Engine([Resource("cube", 10)])
    floor_ready = floor_engine.submit("ready")
    floor_engine.advance(horizon=0)
    floor = InvocationProgram(RecordingBody({"cube": 10}), InvocationTiming(0, 4)).start(
        floor_engine, "floor", floor_ready)
    floor_engine.advance(until=(floor,))
    assert floor_ready.done and floor.time == 4

def test_allocation_backpressure_remains_causal_under_startup():
    engine = Engine([Resource("wire", 10), Resource("cube", 10)], pools={"L1": 64},
                    pool_units={"L1": 1}, trace_limit=100)
    held = Allocation("held", "L1", 64, (0,))
    holder = engine.submit("holder", {"wire": 100}, acquire=(held,), free=(held,))
    own = Allocation("body", "L1", 64, (0,))
    done = InvocationProgram(KernelProgram((
        Stage("service", {"cube": 20}, acquire=(own,), free=(own,)),)),
        InvocationTiming(startup_ms=2)).start(engine, "waiting", engine.submit("ready"))
    engine.advance(until=(done,))
    service = next(row for row in engine.trace if row["task"] == "waiting:body:service")
    assert holder.time == 10 and (service["start_ms"], service["end_ms"]) == (10, 12)
    assert (done.time, engine.serviced, engine.allocated, engine.peak_allocated) == (
        12, {"wire": 100, "cube": 20}, {"L1": 0}, {"L1": 64})

def test_ordinary_and_quantized_classes_pay_inclusive_minima_once():
    raw = hardware_case()
    raw["machine"] = {"backend": "ascend", "sku": SERVICE_SKU, "num_devices": 1}
    mixed = replace(op("mixed", flops=2), vector_ops=1)
    small = operator_costs(raw, (mixed,), input_ready_ms=7)["operators"][0]
    assert all((small["duration_ms"] == pytest.approx(.001237), small["service_before_input"] == {}))
    # A combined Cube/Vector kernel takes one maximum class minimum, not both.
    large = replace(mixed, hbm_bytes=4e8)
    result = graph_costs(raw, (large,), input_ready_ms=7, trace_limit=100)
    assert result["duration_ms"] == pytest.approx(.1 * DRAM_SERVICE_SCALE)
    assert result["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(
        4e8 * DRAM_SERVICE_SCALE)
    assert not any(event["task"].endswith(":startup") for event in result["events"])
    dma = operator_costs(raw, (op("dma", memory=400),))["operators"][0]
    assert dma["duration_ms"] == pytest.approx(1e-7 * DRAM_SERVICE_SCALE)
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    plain = replace(op("quant", flops=2), vector_ops=4, dtype="int8")
    declared = replace(plain, metadata=(("invocation_class", "w8a8_nd_bf16"),))
    empty = replace(op("empty"), recipe="empty", source_boundary="kernel",
                    metadata=(("invocation_class", "empty_aic"),))
    recipes = RecipeBook({"empty": PlannedRecipe(lambda operation, geometry: OperatorPlan(
        Work(), lambda context: KernelProgram(())))})
    for operation, expected in ((plain, .001237), (declared, .003037), (empty, .000439)):
        report = operator_costs(raw, (operation,), recipes=recipes, input_ready_ms=3)
        assert all((report["operator_sum_ms"] == pytest.approx(expected), report["operators"][0]["service_before_input"] == {}))
    report = graph_costs(raw, (replace(declared, hbm_bytes=4e8),), input_ready_ms=3, isolated=False)
    assert report["duration_ms"] == pytest.approx(.1 * DRAM_SERVICE_SCALE)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0

def test_custom_and_runtime_rules_replace_ordinary_assumption():
    raw = hardware_case()
    raw["machine"] = {"backend": "ascend", "sku": SERVICE_SKU, "num_devices": 1,
                      "device": {"kernel_invocations": {
                          "custom_tiny_add": {"minimum_ms": .000876}}}}
    custom = replace(op("tiny"), vector_ops=1,
                     metadata=(("invocation_class", "custom_tiny_add"),))
    assert operator_costs(raw, (custom,))["operators"][0]["duration_ms"] == pytest.approx(.000876)
    disabled = operator_costs(declare(raw, startup=0, minimum=0),
                              (replace(custom, metadata=()),))["operators"][0]
    assert disabled["duration_ms"] < 1e-6
    response_raw = declare(hardware_case(), startup=2)
    with pytest.raises(ValueError, match="duplicates_resource_service_boundary"):
        operator_costs(response_raw, (copy_op(2000),),
                       recipes=RecipeBook({"copy": ResourceService(curve())}))
    response = operator_costs(response_raw, (copy_op(4000),),
                              recipes=RecipeBook({"copy": ResourceService(curve())}))
    # Outside 1000..3000 B the curve has no startup, so the declared 2 ms stays.
    assert response["operators"][0]["duration_ms"] == 6
    assert response["operators"][0]["basis"]["service_models"][0]["queries"] == {
        "analytic_outside_domain": 1}

@pytest.mark.parametrize("minimum,expected", ((2, 7), (8, 8)))
def test_public_operator_graph_and_serving_share_ready_floor(minimum, expected):
    raw = declare(hardware_case(), minimum=minimum)
    raw["workload"]["requests"][0]["arrival_ms"] = 7
    operation = copy_op(4000)
    isolated = operator_costs(raw, (operation,), input_ready_ms=7)["operators"][0]
    graph = graph_costs(raw, (operation,), input_ready_ms=7)
    serving = predict(raw, lower=copy_lower(operation))
    # 4000 B at 1000 B/ms after 3 ms startup; the floor starts at readiness.
    assert isolated["duration_ms"] == graph["duration_ms"] == expected
    assert all((serving["requests"][0]["finish_ms"] == graph["complete_ms"] == 7 + expected, isolated["serviced"] == {"memory@r0:s0": 4000}))
    assert {key: value for key, value in graph["execution"]["serviced"].items() if value} == {
        "memory@r0:s0": 4000}
    assert graph["execution"]["live_tasks"] == serving["execution"]["live_tasks"] == 0
    stored = declare(hardware_case(), startup=3, minimum=8)
    with PrimitiveStore(":memory:") as store:
        store.record(primitive_key(parse_case(stored), operation, "none"), duration_ms=12,
                     run_id="control", repeat_index=0, source="oracle")
        report = graph_costs(stored, (operation,), input_ready_ms=7, primitives=store)
    assert report["complete_ms"] == 19  # 7 + max(12, 8, 3 + 4)
    assert report["isolated"]["operator_sum_ms"] == 12

def test_host_launch_and_graph_replay_stay_outside_kernel_startup():
    raw = declare(hardware_case(), startup=3)
    raw["machine"]["device"]["launch_us"] = 2000
    raw["runtime"]["submission"] = {"graph_replay_ms": 3}
    operations = (op("first", memory=1000), op("second", memory=1000, deps=("first",)))
    device = graph_costs(raw, operations, isolated=False)
    eager = graph_costs(raw, operations, include_submission=True, replay_scope="none",
                        isolated=False, trace_limit=100)
    graph = graph_costs(raw, operations, include_submission=True, replay_scope="full",
                        isolated=False, trace_limit=100)
    # Bodies occupy [5,6] and [9,10]; the second host submission [2,4] overlaps.
    assert (device["duration_ms"], eager["duration_ms"], graph["duration_ms"]) == (8, 10, 11)
    assert all((eager["execution"]["serviced"]["host@r0:s0"] == 4, graph["execution"]["serviced"]["host@r0:s0"] == 3))
    assert sum(row["task"].endswith(":submit") for row in eager["events"]) == 2
    assert all((sum(row["task"].endswith(":submit") for row in graph["events"]) == 1, eager["execution"]["serviced"]["memory@r0:s0"] == 2000))

@pytest.mark.parametrize("minimum,idle_end", ((0, 9), (20, 20)))
def test_idle_collective_receiver_has_startup_without_global_floor(minimum, idle_end):
    raw = declare(cohort_case(2), startup=5, minimum=minimum,
                  match={"recipe": "idle_receiver"})
    graphs = {0: (replace(exchange(0, axis="ep"), recipe="idle_receiver"),),
              1: (exchange(4, axis="ep"),)}
    report = graph_costs(raw, graphs)
    rows = {row["id"]: row for row in report["timeline"]}
    # Rank0's real call enters at 5 ms; four bytes use the link until 9 ms,
    # and only its local floor may keep rank0 until 20 ms.
    assert (rows["r0:exchange"]["complete_ms"], rows["r1:exchange"]["complete_ms"]) == (
        idle_end, 9)
    assert sum(value for key, value in report["execution"]["serviced"].items()
               if key.startswith("link:")) == 4
    assert report["parameters"]["invocation_models"][0]["invocations"] == 1
    assert report["isolated"]["collective_cohorts"][0]["duration_ms"] == idle_end

def test_eager_submissions_pipeline_and_replay_is_one_burst_launch():
    raw = serving_case()
    raw["machine"]["device"]["launch_us"] = 2000
    raw["runtime"]["submission"] = {"scheduler_ms": 100, "completion_ms": 200}
    operations = (op("a", flops=1e6), op("b", memory=1e6, deps=("a",)))
    eager = call_costs(raw, operations, invocations=3, include_submission=True)
    replay = call_costs(raw, operations, invocations=3, mode="graph",
                        include_submission=True, submission={"graph_replay_ms": 3})
    device = call_costs(raw, operations, invocations=3)
    assert (eager["duration_ms"], replay["duration_ms"], device["duration_ms"]) == (13, 9, 6)
    assert tuple(row["execution"]["serviced"]["host@r0:s0"]
                 for row in (eager, replay, device)) == (12, 3, 0)
    with pytest.raises(ValueError, match="outside_callback_boundary"):
        call_costs(raw, operations, include_submission=True,
                   submission={"scheduler_ms": 100, "completion_ms": 200})
    repeated = call_costs(serving_case(), (
        op("producer", memory=2e6), op("consumer", memory=3e6, deps=("producer",))),
        invocations=3, input_ready_ms=7, trace_limit=100)
    assert (repeated["duration_ms"], repeated["complete_ms"], repeated["execution"]["serviced"]["memory@r0:s0"]) == (15, 22, 15e6)


def test_source_call_expansion_precedes_repetition_and_budgeting():
    result = call_costs(serving_case(), (call_op(),), recipes=call_recipe_book(),
                        invocations=2, max_operators=6)
    assert (result["duration_ms"], result["single_call"]["expanded_operators"],
            result["single_call"]["source_boundaries"]["kernel"]) == (6, 3, 2)
    with pytest.raises(ValueError, match="call_expanded_operator_budget_exceeded"):
        call_costs(serving_case(), (call_op(),), recipes=call_recipe_book(),
                   invocations=3, max_operators=6)
    timed = serving_case()
    timed["machine"]["device"]["kernel_invocations"] = {
        "cube": {"minimum_ms": 100}, "vector": {"minimum_ms": 100}}
    charged = operator_costs(timed, (call_op(),), recipes=call_recipe_book())
    assert (charged["operator_sum_ms"],
            sum(model["invocations"] for row in charged["operators"]
                for model in row["basis"]["invocation_models"])) == (200, 2)

def test_independent_kernels_overlap_and_fused_boundary_pays_once():
    raw = timed_case()
    split = (vector("a"), vector("b", deps=("a",)))
    serial = graph_costs(raw, split)
    concurrent = graph_costs(raw, (split[0], vector("b", stream="second")))
    combined = graph_costs(raw, (vector("ab", amount=2),))
    assert all((serial["duration_ms"] == pytest.approx(.005), concurrent["duration_ms"] == combined["duration_ms"] == pytest.approx(.0025)))
    assert serial["execution"]["work"] == combined["execution"]["work"]
    mixed = graph_costs(raw, (replace(vector(), flops=1),))
    assert mixed["duration_ms"] == pytest.approx(.005)  # max, never 2.5+5 us
    assert mixed["parameters"]["invocation_models"][0]["invocations"] == 1
    floor_raw = timed_case()
    floor_raw["workload"]["requests"][0]["arrival_ms"] = 7
    floor_op = vector()
    isolated = operator_costs(floor_raw, (floor_op,), input_ready_ms=7)["operators"][0]
    graph = graph_costs(floor_raw, (floor_op,), input_ready_ms=7)
    serving = predict(floor_raw, lower=copy_lower(floor_op))
    assert (isolated["duration_ms"], graph["complete_ms"],
            serving["requests"][0]["finish_ms"]) == (pytest.approx(.0025), 7.0025, 7.0025)
    assert isolated["serviced"] == {"vector@r0:s0": 1}
    assert operator_costs(floor_raw, (vector(amount=100),))["operators"][0][
        "duration_ms"] == pytest.approx(.1)

def test_no_implicit_aiv_on_dma_or_join_and_per_device_override():
    raw = timed_case()
    operations = (op("dma", memory=1), replace(op("join", deps=("dma",)), family="barrier"))
    result = graph_costs(raw, operations)
    assert all((result["duration_ms"] == pytest.approx(.001), result["parameters"].get("invocation_models", []) == []))
    raw["machine"]["devices"] = [{"kernel_invocations": {"vector": {"minimum_ms": .004}}}]
    machine = Hardware.from_mapping(raw["machine"])
    assert all((Hardware.from_mapping(machine.to_dict()) == machine, dict(machine.dies[0].kernel_invocations)["cube"].minimum_ms == .005))
    assert operator_costs(raw, (vector(),))["operator_sum_ms"] == pytest.approx(.004)
    raw["machine"]["devices"][0]["kernel_invocations"]["vector"] = {"minimum_ms": 0}
    assert operator_costs(raw, (vector(),))["operator_sum_ms"] == pytest.approx(.001)

def test_register_capacity_counts_calls_not_outputs_or_payload():
    # Four calls producing two registers each: outputs are not extra calls.
    elapsed, work = capacity_execute(InstructionInventory(
        {"shuffle": InstructionCount(4, 256)}), {"shuffle": InstructionClass(2, 2)})
    assert elapsed == pytest.approx(4) and work["writes"] == 8 and work["vector"] == 200
    inventory = InstructionInventory({
        "broadcast_load": InstructionCount(4, 256, read_bytes=16),
        "scalar_store": InstructionCount(4, 4, write_bytes=16)})
    elapsed, work = capacity_execute(inventory, {
        "broadcast_load": InstructionClass(1, 1, ub_read=True),
        "scalar_store": InstructionClass(2, ub_write=True)})
    assert all((elapsed == 6 and work["ub"] == 8 and work["stores"] == 4, inventory.to_dict()["instructions"]["broadcast_load"]["read_bytes"] == 16))
    unknown = InstructionInventory({
        "VBitSort32": InstructionCount(3, 96), "VMrgSort4V2": InstructionCount(2, 128)
    }).capacity_work(CONTROL_INSTRUCTIONS)
    assert unknown.resources == {"vbitsort32": 3, "vmrgsort4v2": 2}
    view = PrimitiveInventory(unknown.resources).to_dict({"vbitsort32"})
    assert all((view["priced_service_work"] == {"vbitsort32": 3}, view["unpriced_service_work"] == {"vmrgsort4v2": 2}))

@pytest.mark.parametrize("scale", (.5, 1, 2))
def test_owner_typed_capacity_and_rate_scans_preserve_counts(scale):
    # Eight descriptors on one owner cost 8 us; an idle sibling cannot lend
    # issue capacity, independent owners overlap, same-owner calls contend.
    one, work = service_execute(scale, (0,))
    assert all((one == pytest.approx(.008 / scale), work == {"mte2_descriptor_issue": 8, "vbitsort32": 4}))
    assert all((service_execute(scale, (0, 1))[0] == pytest.approx(one), service_execute(scale, (0, 0))[0] == pytest.approx(2 * one)))

@pytest.mark.parametrize("scale", (.5, 1, 2))
def test_explicit_capacity_transfer_scales_one_owner(scale):
    machine = service_hardware(scale, linked=True)
    assert dict(machine.dies[0].resource_units)["vector"] == 4
    elapsed, work = linked_execution(machine)
    assert all((elapsed == pytest.approx(.008 / scale), work == {"mte2_descriptor_issue": 8}))
    assert Hardware.from_mapping(machine.to_dict()) == machine
    direct = service_hardware(.5, linked=True).to_dict()
    direct["devices"][0]["rates"]["mte2_descriptor_issue"] = 16e6
    assert linked_execution(Hardware.from_mapping(direct))[0] == pytest.approx(.002)
    unlinked = service_hardware(linked=True).to_dict()
    device = unlinked["devices"][0]
    device["rates"].pop("mte2_descriptor_issue", None)
    del device["compute_arch"]["instruction_services"]["mte2_descriptor_issue"]["scale_with"]
    device["rates"]["mte2_aiv"] = 800e9
    assert linked_execution(Hardware.from_mapping(unlinked))[0] == pytest.approx(.008)

def test_declared_scalar_interval_scales_owner_local_span():
    unpriced = grouped_row()
    program = dict(unpriced["operator"]["metadata"])["program_execution"]
    schedule = program["grouped_dequant_schedule"]
    assert (schedule["scalar_service"], schedule["primitive_services"]["unpriced_service_work"]) == (
        "payload_only_issue_latency_not_declared", {"scalar_gm_access": 144})
    assert "scalar_gm_access@r0:s0" not in unpriced["serviced"]
    spans, payloads, schedules = [], [], []
    for factor in (.5, 1, 2):
        row = grouped_row(scalar_gm_access=1000 / factor)
        schedules.append(dict(row["operator"]["metadata"])["program_execution"]["grouped_dequant_schedule"])
        assert row["serviced"]["scalar_gm_access@r0:s0"] == pytest.approx(144)
        spans.append(row["duration_ms"])
        payloads.append({key: value for key, value in row["serviced"].items() if "scalar_gm_access" not in key})
    assert all((schedules[0]["launched_aiv"] == 9, schedules[0]["rows_per_owner"] == (15,) * 8 + (0,)))
    assert schedules[0]["primitive_services"]["priced_service_work"] == {
        "scalar_gm_access": 144}
    # Each active owner runs 16 serialized PIPE_S calls at 1000/factor ns.
    assert all((spans[1] - spans[2] == pytest.approx(.008), spans[0] - spans[1] == pytest.approx(.016)))
    assert payloads[0] == payloads[1] == payloads[2]

def test_explicit_enqueue_overrides_device_cost_and_runs_ahead():
    raw = serving_case()
    raw["machine"]["device"]["launch_us"] = 1000
    raw["runtime"]["submission"] = {"kernel_launch_ms": .25}
    operations = (op("a", flops=4e6), op("b", flops=2e6, deps=("a",)))
    report = graph_costs(raw, operations, include_submission=True, trace_limit=100)
    # b is enqueued at .5 while a computes from .25 to 4.25; charging the
    # enqueue after device completion would incorrectly give 6.5.
    assert report["duration_ms"] == 6.25
    summary = report["execution"]["submission"]
    assert all((summary["host_service_ms"] == .5, summary["eager_invocations"] == 2 and summary["graph_replays"] == 0))
    assert [event["end_ms"] for event in report["events"] if event["task"].endswith(":submit")] == [.25, .5]
    raw["runtime"]["submission"]["kernel_launch_ms"] = 0
    excluded = graph_costs(raw, operations, include_submission=True)
    assert all((excluded["duration_ms"] == 6, excluded["execution"]["submission"]["host_service_ms"] == 0))
    del raw["runtime"]["submission"]["kernel_launch_ms"]
    legacy = graph_costs(raw, operations, include_submission=True)
    assert all((legacy["duration_ms"] == 7, legacy["execution"]["submission"]["host_service_ms"] == 2))

def test_capture_charges_one_replay_per_segment_and_preserves_work():
    raw = serving_case(pp=2)
    raw["machine"]["device"]["launch_us"] = 1000
    raw["runtime"].update(graph_mode="auto", graph_capture_keys=[4],
                          graph_capture_mode="full")
    raw["runtime"]["submission"] = {"kernel_launch_ms": .25, "graph_replay_ms": .1}
    graphs = target_graphs(raw, phase="decode", tokens=1)
    captured = graph_costs(raw, graphs, include_submission=True)
    eager = graph_costs(raw, graphs, include_submission=True, replay_scope="none")

    summary = captured["execution"]["submission"]
    assert all((summary["eager_invocations"] == 0 and summary["graph_replays"] == 2, summary["host_service_ms"] == pytest.approx(.2)))
    assert all((eager["execution"]["submission"]["eager_invocations"] > 2, captured["duration_ms"] < eager["duration_ms"]))
    assert captured["execution"]["work"] == eager["execution"]["work"]

def test_pd_pools_keep_independent_enqueue_costs():
    raw = pipeline_case()
    raw["pools"]["prefill"]["runtime"]["submission"] = {"kernel_launch_ms": .25}
    raw["pools"]["decode"]["runtime"]["submission"] = {"kernel_launch_ms": .5}
    report = predict(raw, lower=linear_work)
    assert all((report["requests"][0]["finish_ms"] == 9.25, report["execution"]["handoff_bytes"] == 384))
    prefill, decode = (report["execution"]["submission"][pool] for pool in ("prefill", "decode"))
    assert (prefill["host_service_ms"], decode["host_service_ms"]) == (.25, 1)
    assert (prefill["eager_invocations"], decode["eager_invocations"]) == (1, 2)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    api = predict(serving_case(prompt=4, output=1), lower=linear_work,
                  submission=SubmissionPolicy(kernel_launch_ms=.25))
    stage = api["execution"]["submission"]["stages"][0]
    assert api["metrics"]["makespan_ms"] == 4.25
    assert (stage["kernel_launch_ms"], stage["kernel_parameter"]) == (
        .25, "api.submission.kernel_launch_ms")
    api = predict(serving_case(prompt=4, output=1), lower=linear_work,
                  submission=SubmissionPolicy(kernel_launch_ms=.25))
    stage = api["execution"]["submission"]["stages"][0]
    assert api["metrics"]["makespan_ms"] == 4.25
    assert (stage["kernel_launch_ms"], stage["kernel_parameter"]) == (
        .25, "api.submission.kernel_launch_ms")

def test_device_only_async_execution_does_not_charge_scheduler():
    raw = serving_case()
    raw["runtime"].update(async_scheduling=True, submission={
        "scheduler_ms": 5, "kernel_launch_ms": 2, "completion_ms": 3})
    parsed = parse_case(raw)
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    step = execution.add(linear_work(
        IterationPlan((RequestView("r", 4, 0, True, "prefill"),)),
        parsed.model, parsed.runtime, parsed.machine), 0, 0)
    execution.prepare_next(0)
    execution.engine.advance()
    assert step.completion.time == execution.engine.now == 4
    summary = execution.summary()
    assert all((summary["serviced"]["scheduler@r0"] == 0, summary["submission"]["host_service_ms"] == 0))
    assert all((summary["submission"]["eager_invocations"] == 0, summary["live_tasks"] == summary["live_programs"] == 0))
