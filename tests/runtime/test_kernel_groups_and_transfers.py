"""Native kernel groups, finite transfers, and detailed pipeline equivalence.

[Role] Exercise phase DAGs, physical owners, credits, and pipeline scheduling.
[Invariants] Readiness, resource work, and allocation lifetimes survive lowering
  and template expansion. Independent recurrences account for every transfer.
[Timeouts & Locks] Bounded in-memory simulations, with no threads or locks.
[Side-Effect Policy] Read-only example case data and scoped monkeypatches.
"""
from dataclasses import replace
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim import Case, graph_costs, operator_costs
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction import scheduling
from mindie_sim.prediction.cost import CostModel, primitive_key
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.instructions import REGISTER_RESOURCES
from mindie_sim.prediction.matrix import TiledMatmul
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.operators import gdn_output, gdn_prepare, gdn_state
from mindie_sim.prediction.operators.mixed_core import phase_stages
from mindie_sim.prediction.primitives import PrimitiveStore
from mindie_sim.prediction.program import (
    KernelProgram, PipelinePhase, PipelineProgram, PipelineTemplate,
    PipelineTemplateProgram, RecipeBook, RecipeContext, Stage,
    indexed_pipeline_consumer_available,
)
from mindie_sim.prediction.services import service_identity
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import TransportOptions
from mindie_sim.prediction.transfer import FiniteTransfer
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan

from _support.collective_cases import cohort_case, exchange
from _support.hardware import hardware_case
from _support.kernel_invocation import BUFFER, competitor, declare, fork, group, op, phase, raw_case, recipes
from tools.finite_transfer_oracle import finite_transfer, homogeneous_duration

pytestmark = pytest.mark.tier0

_ATTENTION_CASE = Path("examples/cases/qwen38-a5.json")

def _simulate(amount, slots, response=4, rate=10, granule=10, *, trace=0):
    engine = Engine([Resource("wire", rate, "bytes")], trace_limit=trace)
    done = FiniteTransfer({"wire": amount}, granule, slots, response).start(engine, "copy", engine.submit("ready"))
    engine.advance(until=(done,))
    return done.time, engine

@pytest.mark.parametrize("minimum,span", [(0, 9), (8, 9), (12, 12)])
def test_phase_fork_retains_overlap_and_inclusive_one_kernel_minimum(minimum, span):
    raw = declare(hardware_case(), startup=3, minimum=minimum,
                  match={"kernel": "NativeDispatch"})
    graph = graph_costs(raw, fork(dependency=("input",)), input_ready_ms=7,
                        external_inputs={"input": 9}, trace_limit=200)
    assert graph["complete_ms"] == 9 + span
    timeline = {row["op_id"]: row for row in graph["timeline"]}
    assert timeline["dispatch.count"]["ready_ms"] == timeline["dispatch.payload"]["ready_ms"] == 13
    assert {name: count for name, count in graph["execution"]["serviced"].items() if count} == {
        "memory@r0:s0": 5000, "vector@r0:s0": 3000}
    assert all(event["start_ms"] >= 12 for event in graph["events"] if event["demand"])
    assert graph["execution"]["live_tasks"] == graph["execution"]["live_programs"] == 0
    assert operator_costs(raw, fork())["operator_sum_ms"] == 8

@pytest.mark.parametrize("scope,hosts,span", [("none", 2, 22), ("full", 1, 23)])
def test_groups_share_one_enqueue_each_and_one_capture_replay(scope, hosts, span):
    raw = declare(hardware_case(), startup=3, minimum=10,
                  match={"kernel": "NativeDispatch"})
    raw["machine"]["device"]["launch_us"] = 2000
    raw["runtime"]["submission"] = {"graph_replay_ms": 3}
    ops = fork("first") + fork("second", ("first.exit",))
    graph = graph_costs(raw, ops, replay_scope=scope, include_submission=True, isolated=False)
    submission = graph["execution"]["submission"]
    assert graph["duration_ms"] == span and submission["host_service_ms"] == (4 if scope == "none" else 3)
    assert (submission["eager_invocations"], submission["graph_replays"]) == ((hosts, 0) if scope == "none" else (0, hosts))

def test_interleaved_phases_keep_kernel_order_and_wait_for_every_input():
    first, second = fork("first"), fork("second")
    ops = (first[0], second[0], second[1], first[1], second[2], first[2], first[3], second[3])
    assert graph_costs(hardware_case(), ops, isolated=False)["duration_ms"] == 12
    ops = list(fork(dependency=("early",)))
    ops[2] = replace(ops[2], deps=("dispatch.entry", "late"))
    raw = declare(hardware_case(), match={"kernel": "NativeDispatch"})
    result = graph_costs(raw, ops, external_inputs={"early": 2, "late": 11}, trace_limit=200)
    assert result["complete_ms"] == 20
    assert all(event["start_ms"] >= 14 for event in result["events"] if event["demand"])

@pytest.mark.parametrize("minimum,span", [(0, 24), (10, 30)])
def test_prior_branch_stream_tail_is_part_of_whole_invocation_readiness(minimum, span):
    raw = declare(hardware_case(), startup=3, minimum=minimum,
                  match={"kernel": "NativeDispatch"})
    operations = (
        op("prior_payload_producer", memory=20000, stream="payload"),
        phase("entry", entry=True),
        phase("payload", memory=1000, stream="payload", deps=("entry",)),
        phase("exit", exit=True, deps=("payload",)),
    )
    result = graph_costs(raw, operations, isolated=False, trace_limit=200)
    timeline = {row["op_id"]: row for row in result["timeline"]}
    assert timeline["prior_payload_producer"]["complete_ms"] == 20
    assert timeline["entry"]["ready_ms"] == timeline["payload"]["ready_ms"] == 23
    assert result["complete_ms"] == span

def test_source_plans_are_resource_phases_but_whole_group_uses_mixed_hardware_class():
    raw = hardware_case()
    raw["machine"]["device"]["kernel_invocations"] = {
        "cube": {"startup_ms": 3, "minimum_ms": 10},
        "vector": {"startup_ms": 2, "minimum_ms": 8},
    }
    boundaries = []

    def planner(operation, geometry):
        work = Work(matrix={"bf16": 2000}) if operation.op_id == "matrix" else Work(resources={"vector": 1000})

        def lower(context):
            boundaries.append(context.service_boundary)
            assert context.invocation_timing is None
            return KernelProgram((Stage("service", context.bind(work)),))
        return OperatorPlan(work, lower)
    operations = (
        replace(phase("vector", entry=True), recipe="native_phase", source_boundary="kernel"),
        replace(phase("matrix", exit=True, deps=("vector",)), recipe="native_phase", source_boundary="kernel"),
    )
    recipes = RecipeBook({"native_phase": PlannedRecipe(planner)})
    result = graph_costs(raw, operations, recipes=recipes, isolated=False)
    assert result["duration_ms"] == 10 and boundaries == ["resource_phase"] * 2
    assert tuple(result["execution"]["serviced"][f"{r}@r0:s0"] for r in ("matrix", "vector")) == (1, 1000)

def test_full_kernel_primitive_cannot_leak_into_a_phase_and_external_edges_use_exit():
    raw, operations = hardware_case(), fork()
    bare, parsed = replace(operations[0], metadata=()), parse_case(raw)
    assert primitive_key(parsed, operations[0], "none").to_dict()["scope"] == "resource_phase"
    with PrimitiveStore(":memory:") as store:
        store.record(primitive_key(parsed, bare, "none"), duration_ms=100,
                     run_id="independent-kernel", repeat_index=0, source="boundary-control")
        report = graph_costs(raw, operations, primitives=store)
    assert report["duration_ms"] == 6
    with pytest.raises(ValueError, match="external_dependency_requires_exit"):
        graph_costs(raw, operations + (op("wrong_consumer", deps=("dispatch.count",)),))

def _transport_group(volumes, timed):
    entry = replace(phase("entry", entry=True, memory=1000), vector_ops=1000) if timed else phase("entry", entry=True)
    transfers = []
    for name, volume in volumes:
        transfer = exchange(volume, axis="ep")
        transfers.append(replace(transfer, op_id=name, deps=("entry",), stream=name,
                                 metadata=transfer.metadata + phase(name).metadata))
    return (entry, *transfers, phase("exit", exit=True, deps=tuple(name for name, _ in volumes)))

@pytest.mark.parametrize("timing,volumes", [("runtime", (4, 4)), ("hardware", (4, 4)), ("runtime", (0, 4)), ("runtime", (0, 0))])
def test_transport_only_phases_preserve_shared_invocation_and_rank_bytes(timing, volumes):
    active = volumes[0] > 0
    raw = declare(cohort_case(2), startup=3, minimum=10, match={"kernel": "NativeDispatch"})
    raw["runtime"]["kernels"]["families"]["alltoall"] = {"kind": "direct_peer_collective", "max_inflight_per_rank": 1}
    if active:
        raw["machine"]["device"]["launch_us"] = 2000
    if timing == "hardware":
        raw["runtime"]["kernels"].pop("invocations")
        raw["machine"]["device"]["kernel_invocations"] = {"vector": {"startup_ms": 3, "minimum_ms": 10}}
    graphs = {rank: _transport_group(((("count", 2),) if active else ()) + (("payload", v),), active)
              for rank, v in enumerate(volumes)}
    report = graph_costs(raw, graphs, include_submission=active, isolated=active, trace_limit=200)
    assert report["duration_ms"] == pytest.approx(12 if active else 10)
    execution = report["execution"]
    assert sum(v for k, v in execution["serviced"].items() if k.startswith("link:")) == sum(volumes) + 4 * active
    assert execution["live_tasks"] == execution["live_programs"] == 0
    if active:
        submission = execution["submission"]
        assert (submission["eager_invocations"], submission["host_service_ms"]) == (2, 4)
    else:
        timeline = {row["id"]: row for row in report["timeline"]}
        for rank in (0, 1):
            assert (timeline[f"r{rank}:entry"]["ready_ms"], timeline[f"r{rank}:exit"]["complete_ms"]) == (3, 10)
        assert sum(row["invocations"] for row in report["parameters"]["invocation_models"]) == 2
        raw["runtime"]["kernels"].pop("invocations")
        raw["machine"]["device"]["kernel_invocations"] = {"vector": {"startup_ms": 30, "minimum_ms": 100}}
        report = graph_costs(raw, graphs, isolated=False)
        assert report["duration_ms"] == max(volumes) and not report["parameters"].get("invocation_models")

@pytest.mark.parametrize("memory_rate,expected", [(1e9, 1), (100, 3)])
def test_split_core_reads_use_independent_pipes_but_shared_memory(memory_rate, expected):
    engine = Engine([
        Resource("memory", memory_rate),
        Resource("aic_read", 100, units=1),
        Resource("aiv_read", 200, units=2),
    ], trace_limit=100)
    context = RecipeContext(
        None, {}, {"hbm": "memory", "mte2_aic": "aic_read", "mte2_aiv": "aiv_read"},
        {name: resource.rate for name, resource in engine.resources.items()}, 1,
        resource_units={"mte2_aic": 1, "mte2_aiv": 2},
    )
    work = Work(resources={"hbm": 100, "mte2": 100})
    stages = tuple(stage for name, lane in (("cube", None), ("vec0", 0), ("vec1", 1))
                   for stage in phase_stages(name, work, context, 0, aiv=lane))
    end = KernelProgram(stages).start(engine, "mixed", engine.submit("ready"))
    engine.advance(until=(end,))
    assert end.time == pytest.approx(expected)
    assert engine.serviced == {"memory": 300, "aic_read": 100, "aiv_read": 200}
    loads = {row["task"]: row for row in engine.trace if row["task"].endswith(":load")}
    assert {name: row["placement"] for name, row in loads.items()} == {
        "mixed:cube:load": {"aic_read": (0,)}, "mixed:vec0:load": {"aiv_read": (0,)}, "mixed:vec1:load": {"aiv_read": (1,)}}

@pytest.mark.parametrize("cores", [1, 4])
@pytest.mark.parametrize("role", ["reg_issue_slots", "ub_copy"])
def test_source_vector_modes_cannot_borrow_other_aiv_owners(cores, role):
    engine = Engine([Resource("source_pipe", 200 * cores, units=2 * cores)], trace_limit=20)
    context = RecipeContext(
        None, {}, {"custom_vector_pipe": "source_pipe"},
        {"source_pipe": 200 * cores}, cores,
        resource_units={"custom_vector_pipe": 2 * cores},
        service_modes={role: ("custom_vector_pipe", 200000 * cores)},
    )
    program = KernelProgram(phase_stages(
        "vector", Work(resources={role: 100}), context, 0, aiv=1))
    done = program.start(engine, "mixed", engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == pytest.approx(1)
    assert engine.serviced == {"source_pipe": pytest.approx(100)}
    compute = next(row for row in engine.trace if row["task"] == "mixed:vector:compute")
    assert compute["placement"] == {"source_pipe": (1,)}

@pytest.mark.parametrize("module", [gdn_prepare, gdn_state, gdn_output])
def test_native_source_plan_accounts_for_every_split_read_without_generic_pipe(module):
    raw = hardware_case()
    raw.pop("runtime")
    reads = ("mte2_aic", "mte2_aiv")
    roles = ("hbm", "vector", "mte1", "fixpipe", *reads, "mte3", *REGISTER_RESOURCES)
    raw["machine"]["device"].update(
        compute_units=2, compute_peaks={"bf16": 1e12, "fp32": 1e11},
        rates=dict.fromkeys(roles, 1e11),
        resource_units={**dict.fromkeys(("vector", "mte2_aiv", "mte3", *REGISTER_RESOURCES), 4),
                        "mte2_aic": 2},
        service_modes={role: {"resource": "vector", "rate": 1e11}
                       for role in ("ub_copy", "reg_issue_slots")},
    )
    operation = module.make_op({"lengths": [73], "heads": 3, "key_heads": 1,
                                "dk": 128, "dv": 128, "chunk": 64,
                                "dtype": "bf16", "gate_dtype": "fp32", "use_exp2": True})
    row, = operator_costs(Case.from_mapping(raw), (operation,), recipes=RecipeBook(
        {operation.recipe: PlannedRecipe(module.build)}))["operators"]
    for role in reads:
        key = role + "@r0:s0"
        assert row["declared_service"][key] > 0
        assert row["serviced"][key] == pytest.approx(row["declared_service"][key])
    assert "mte2@r0:s0" not in row["serviced"] and row["live_tasks"] == 0

@pytest.mark.parametrize("mode,expected,bytes_", [
    ("credit_wait", (10, 4), {"wire": 50}), ("shared", (12, 12), {"wire": 40}),
    ("independent", (15,), {"a": 50, "b": 30}), ("fifo", (6, 2), {"wire": 60}),
])
def test_transfers_share_service_and_keep_credit_windows_local(mode, expected, bytes_):
    resources = (Resource("a", 10, "bytes"), Resource("b", 5, "bytes")) if mode == "independent" else (
        Resource("wire", 10, "bytes", policy="fifo" if mode == "fifo" else "fair"),)
    engine = Engine(resources, trace_limit=100 if mode == "credit_wait" else 0)
    ready = engine.submit("ready")
    demand = {"a": 50, "b": 25} if mode == "independent" else {"wire": 50 if mode == "fifo" else 20}
    ends = [FiniteTransfer(demand, 10, 2 if mode == "independent" else 1, 0 if mode == "fifo" else 4)
            .start(engine, "copy", ready)]
    if mode == "credit_wait":
        ends.append(engine.submit("other", {"wire": 30}, release=1))
    if mode == "shared":
        ends.append(FiniteTransfer(demand, 10, 1, 4).start(engine, "other", ready))
    if mode == "fifo":
        engine.advance(horizon=.5)
        ends.append(engine.submit("other", {"wire": 10}))
    engine.advance(until=ends, first=False)
    assert tuple(end.time for end in ends) == expected and engine.serviced == bytes_
    if mode == "credit_wait":
        service = [row for row in engine.trace if row["task"].endswith(":service") and row["demand"]]
        assert [(row["start_ms"], row["end_ms"]) for row in service] == [(0, 1), (5, 6)]

def test_live_queue_is_bounded_independently_of_transfer_length():
    _, short = _simulate(100, 3)
    _, long = _simulate(20_000, 3)
    assert long.peak_tasks == short.peak_tasks <= 2 * 3 + 2
    assert long.summary()["live_tasks"] == long.summary()["live_programs"] == 0

def test_engine_matches_independent_serial_server_credit_recurrence():
    for payload, slots, rate, response in product(
        (1, 10, 41, 100, 231), (1, 2, 3, 32), (2, 10, 100), (0, .3, 4),
    ):
        expected = finite_transfer(payload, 10, rate, slots, response)
        actual, engine = _simulate(payload, slots, response, rate)
        assert actual == pytest.approx(expected["duration_ms"])
        assert engine.serviced["wire"] == expected["charged_bytes"]


def test_credit_window_includes_the_last_response_and_rounded_tail():
    result = finite_transfer(50, 10, 10, 2, 4)
    assert (result["duration_ms"], result["credit_stall_ms"], result["serialization_ms"]) == (15, 6, 5)
    duration, engine = _simulate(50, 2, trace=100)
    assert duration == 15
    assert [row["end_ms"] for row in engine.trace if row["task"].endswith(":response")] == [5, 6, 10, 11, 15]
    for payload, slots, delay, expected, charged in (
        (50, 1, 4, 25, 50), (50, 8, 4, 9, 50), (50, None, 4, 9, 50),
        (10, 1, 4, 5, 10), (10, 8, 4, 5, 10), (40, 2, 4, 11, 40),
        (41, 2, 4, 15, 50), *((41, slots, 0, 5, 50) for slots in (1, 2, None)),
    ):
        actual, engine = _simulate(payload, slots, response=delay)
        assert (actual, engine.serviced["wire"]) == (expected, charged)

def test_partial_transaction_has_its_own_byte_boundary():
    rounded, exact = (finite_transfer(41, 10, 10, 2, 4, rounded_tail=rounded) for rounded in (True, False))
    assert rounded["transactions"] == exact["transactions"] == 5
    assert (rounded["charged_bytes"], exact["charged_bytes"]) == (50, 41)
    assert rounded["duration_ms"] == 15
    assert (exact["duration_ms"], exact["credit_stall_ms"]) == pytest.approx((14.1, 6))

def test_recurrence_matches_closed_form_across_credit_and_bandwidth_transitions():
    for count, service, credits, response in product(
        (0, 1, 2, 5, 17), (.125, 1, 8), (1, 2, 5), (0, .5, 4, 20),
    ):
        result = finite_transfer(count * 16, 16, 16 / service, credits, response)
        expected = homogeneous_duration(count, service, credits, response)
        assert result["duration_ms"] == pytest.approx(expected)
        assert result["charged_bytes"] == count * 16

@pytest.mark.parametrize("buffered", [False, True])
def test_invocation_identity_and_retirement_are_scoped_per_iteration(buffered):
    raw = raw_case() if buffered else declare(hardware_case(), startup=3, minimum=12,
                                                  match={"kernel": "NativeDispatch"})
    parsed = parse_case(raw)
    execution = Execution(parsed, CostModel(parsed), recipes=recipes() if buffered else None,
                          include_submission=False)
    work = IterationWork(group() if buffered else fork(), IterationPlan(()), 1, 0, 0, None, "decode", "group")
    first, second = (execution.add(work, 0, 5) for _ in range(2))
    execution.engine.advance(until=(first.completion, second.completion), first=False)
    assert (first.completion.time, second.completion.time) == ((8, 11) if buffered else (17, 29))
    assert not execution.engine.allocations and not any(execution.engine.allocated.values())
    assert execution.engine.summary()["live_tasks"] == execution.engine.summary()["live_programs"] == 0
    if not buffered:
        assert next(iter(execution.cost.invocations.values()))["invocations"] == 2
        assert graph_costs(raw, fork() + (op("later", memory=1000, stream="count"),))["complete_ms"] == 13

def _event(report, suffix, contains=""):
    return next(row for row in report["events"] if row["task"].endswith(suffix) and contains in row["task"])

@pytest.mark.parametrize("owner,units,startup,floor,prior,start,finish", [
    (0, [0], 0, 0, False, 3, 4), (1, [0], 0, 0, False, .5, 3),
    (1, [0, 1], 0, 8, False, 8, 9), (0, [0], 2, 8, True, 5, 13), (1, [0, 1], 0, 0, True, 5, 8),
])
def test_kernel_buffer_reserves_all_declared_owners_from_admission_through_retirement(
    owner, units, startup, floor, prior, start, finish,
):
    entry, *body = group()
    scratch, = dict(entry.metadata)["device_kernel_group"]["buffers"]
    assert scratch["id"] == BUFFER
    scratch["units"] = units
    rival = competitor(owner)
    ops = (replace(rival, op_id="prior", deps=(), vector_ops=5000), replace(entry, deps=("begin",)), *body) if prior else (entry, *body, rival)
    result = graph_costs(raw_case(startup=startup, minimum=floor), ops, recipes=recipes(),
                         external_inputs={"begin": 1, "late": .5}, isolated=False, trace_limit=100)
    assert _event(result, ":buffers_acquire")["start_ms"] == (start if prior else 0)
    if prior:
        assert _event(result, ":service", ".first:")["start_ms"] == start + startup
    else:
        service = _event(result, ":service", "competitor")
        assert (service["start_ms"], service["end_ms"]) == (start, start + 1)
        assert _event(result, ":retire")["end_ms"] == max(3, floor)
    assert result["complete_ms"] == finish and not any(result["execution"]["live_allocated"].values())
    assert result["execution"]["serviced"]["vector@r0:s0"] == (8000 if prior else 4000)

def test_buffer_admission_waits_for_all_inputs_and_isolated_phases_reserve_locally():
    ops = list(group())
    ops[2] = replace(ops[2], deps=(*ops[2].deps, "late"))
    result = graph_costs(raw_case(), ops, recipes=recipes(), external_inputs={"late": 11},
                         isolated=False, trace_limit=100)
    assert (_event(result, ":buffers_acquire")["start_ms"], result["complete_ms"]) == (11, 14)
    isolated, = operator_costs(raw_case(), (ops[1],), recipes=recipes())["operators"]
    assert isolated["duration_ms"] == 1 and isolated["peak_buffers"]["ub@r0:s0"] == 1024

def _tile(index, *, size=1, compute=1, resident=(), retire=()):
    panel = Allocation(f"buf{index}", "p", size, (0,))
    return (Stage("load", {"r": 1}, acquire=(panel, *resident)),
            Stage("compute", {"r": compute}, after=("load",), free=(panel, *retire)))

def _pipeline_engine(capacity=2):
    return Engine([Resource("r", 1)], pools={"p": capacity}, pool_units={"p": 1})

def _binding(name, index=0, scope="panel"):
    return dict(name=name, scope=scope, owner=0, ordinal=index, pool="p", size=1,
                units=(0,), action="acquire", phase=index, repeat=1)

@pytest.mark.parametrize("form", ["detailed", "bindings", "template", "motif"])
@pytest.mark.parametrize("external,time,peak", [("work", 20, 2), ("hold", 16, 2), ("tail", 2, 4)])
def test_pipeline_preserves_fair_competition_and_slot_availability(form, external, time, peak):
    tail = external == "tail"
    engine = _pipeline_engine(10 if tail else 2)
    tile = (lambda i: _tile(i, size=1 + 2 * i, compute=0)) if tail else _tile
    program = PipelineProgram(2 if tail else 8, 2, tile, homogeneous=True)
    if external == "hold":
        held = Allocation("external", "p", 1, (0,))
        engine.submit("external-hold", acquire=(held,))
    ready = engine.submit("ready")
    if form == "detailed":
        done = program.start(engine, "p", ready)
    elif form == "bindings":
        bindings = () if tail else tuple(_binding(f"buf{i}", i) for i in range(program.count))
        done = program.start_template(engine, "p", ready, bindings=bindings)
    else:
        panels = 4 if form == "motif" and not tail else 1
        template = program.template(panels_per_output=panels)
        if panels == 4:
            assert [(p.kind, p.count) for p in template.phases] == [("first", 1), ("middle", 2), ("last", 1)] * 2
        done = scheduling.start_pipeline_template(engine, template, "p", ready)
    if external == "work":
        work = engine.submit("external", {"r": 4})
        successor = engine.submit("successor", after=(work,))
    elif external == "hold":
        engine.submit("external-release", after=(done,), free=(held,))
    engine.advance()
    assert done.time == time == engine.serviced["r"]
    assert engine.peak_allocated["p"] == peak and not engine.allocated["p"]
    if external == "work":
        assert work.time == successor.time == 11

@pytest.mark.parametrize("count,quiescent,explicit,instances", [
    (0, False, False, 1), (1, False, True, 1), (2, True, False, 1), (1, False, False, 2),
])
def test_pipeline_setup_and_drain_share_the_invocation_allocation(count, quiescent, explicit, instances):
    scratch = Allocation("w", "p", 1, (0,))
    template = PipelineTemplate(0, 1, (PipelinePhase("body", (Stage("work", {"r": 1}),), count),) if count else (),
                                setup=(Stage("reserve", acquire=(scratch,)),),
                                drain=(Stage("release", {"r": 1}, free=(scratch,)),),
                                bindings=(_binding("w", scope="invocation"),) if explicit else ())
    engine = _pipeline_engine(instances)
    ready = engine.submit("ready")
    ends = [(PipelineTemplateProgram(template).start(engine, "p", ready) if explicit else
             scheduling.start_pipeline_template(engine, template, f"outer:tile{i}", ready, quiescent=quiescent))
            for i in range(instances)]
    engine.advance(until=ends, first=False)
    assert [done.time for done in ends] == [engine.serviced["r"]] * instances
    assert engine.serviced["r"] == instances * (count + 1)
    assert engine.peak_allocated["p"] == instances and not engine.allocated["p"] and not engine._periodic_runs

def test_accumulator_stays_resident_across_panel_repeats():
    accumulator = Allocation("accumulator", "p", 1, (0,))
    template = PipelineTemplate(0, 2, (
        PipelinePhase("first", _tile(0, resident=(accumulator,)), 1),
        PipelinePhase("middle", _tile(1), 3),
        PipelinePhase("last", _tile(4, retire=(accumulator,)), 1),
    ))
    engine = _pipeline_engine(4)
    done = scheduling.start_pipeline_template(engine, template, "m", engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == 10 and engine.peak_allocated["p"] == 3 and not engine.allocated["p"]

@pytest.mark.parametrize("quiescent", [False, True])
def test_long_pipeline_reuses_prepared_stages_and_bounds_live_tasks(quiescent, monkeypatch):
    calls, prepare = [], Stage.prepare

    def counting(self, *args):
        calls.append(self.name)
        return prepare(self, *args)

    monkeypatch.setattr(Stage, "prepare", counting)
    engine = _pipeline_engine()
    template = PipelineProgram(65, 2, lambda _: (Stage("work", {"r": 1}),),
                               setup_ms=1, homogeneous=True).template()
    done = scheduling.start_pipeline_template(engine, template, "p", engine.submit("ready"), quiescent=quiescent)
    engine.advance(until=(done,))
    assert done.time == 66 and engine.serviced["r"] == 65
    assert calls.count("work") == 1 and engine.peak_tasks < 10 and engine.summary()["live_tasks"] == 0
    if quiescent:
        assert engine.completed_count <= 10

def _context(m=67, n=35, k=129):
    operation = gemm_op(op_id="matrix", deps=(), stage=0, layer=0, requests=(),
                        m=m, n=n, k=k, quant=QuantSpec("bf16"), category="dense_mlp", phase="prefill")
    rates = {"cube": 128, "hbm": 1024, "mte2": 16, "mte1": 1024, "mte3": 1024}
    pools = {"l1": 4096, "l0c": 4096}
    return RecipeContext(operation, {"cube": operation.flops}, {r: r for r in rates}, rates, 2,
                         pool_names={p: p for p in pools}, pool_capacities=pools, pool_units=dict.fromkeys(pools, 2),
                         resource_units={r: 1 if r == "hbm" else 2 for r in rates})

def _matrix_engine(context):
    return Engine([Resource(r, rate, units=context.resource_units[r]) for r, rate in context.rates.items()],
                  pools=context.pool_capacities, pool_units=context.pool_units, trace_limit=256)

def test_indexed_matmul_matches_detailed_schedule_and_scopes_panel_and_output_buffers(monkeypatch):
    context = _context()
    results = []
    for indexed in (False, True):
        engine = _matrix_engine(context)
        program = TiledMatmul(32, 32, 32, slots=2, template_execution=indexed).lower(context)
        done = program.start(engine, "m", engine.submit("ready"))
        engine.advance(until=(done,))
        results.append((done.time, engine.serviced, engine.peak_allocated))
    assert results[0] == results[1]
    template, other = TiledMatmul(32, 32, 32, slots=2).pipeline_templates(context)
    assert (template.owner, other.owner, template.tiles, other.tiles) == (0, 1, 15, 15)
    assert [template.index_of(i) for i in (0, 1, 4, 5)] == [
        (0, 1, 0, 0, 0), (1, 4, 0, 1, 1), (2, 5, 0, 4, 0), (0, 6, 1, 0, 1)]
    assert {row["name"]: row["scope"] for row in template.allocation_bindings} == {"accumulator": "output", "panel": "panel"}
    engine = _matrix_engine(context)
    done = PipelineTemplateProgram(template).start(engine, "m", engine.submit("ready"))
    engine.advance(until=(done,))
    granted, freed = ({row["name"] for row in engine.allocation_trace if row["action"] == action}
                      for action in ("acquire", "release"))
    assert granted == freed == ({f"m:owner0:output{i}:accumulator" for i in range(3)} |
                                {f"m:owner0:panel{i}:panel" for i in range(15)})
    assert not any(engine.allocated.values())
    monkeypatch.delattr(scheduling.start_pipeline_template, "indexed_panels_abi")
    assert not indexed_pipeline_consumer_available()
    expanded = TiledMatmul(32, 32, 32, slots=2).pipeline_templates(context)
    assert [(t.tiles, t.motif_repeats, t.index_of) for t in expanded] == [(15, 3, None)] * 2
    ready = engine.submit("blocked")
    before = engine.summary()
    with pytest.raises(ModelingError, match="pipeline_indexed_consumer_unavailable"):
        PipelineTemplateProgram(template).start(engine, "blocked", ready)
    assert engine.summary() == before

@pytest.mark.parametrize("mode,program", [("decode", "StagedGQADecode"), ("prefill", "StagedGQAPrefill")])
def test_attention_pipeline_conserves_declared_memory_through_recipe_selection(mode, program):
    raw = json.loads(_ATTENTION_CASE.read_text())
    if mode == "decode":
        raw["runtime"]["attention_policy"] = {"kv_storage": "paged", "fused_batch": False}
        ops = target_operators(raw, phase=mode, tokens=1, context=1028)
        options = {"input_ready_ms": 7}
    else:
        ops = target_operators(raw, phase=mode, tokens=256, context=512, sequences=2)
        options = {"recipes": RecipeBook.from_mapping(
            {"recipes": {"flash_attention": {"kind": "staged_gqa_prefill"}}})}
        index = next(i for i, operation in enumerate(ops) if operation.family == "attention")
        ops = ops[index:index + 1]
    row = next(row for row in operator_costs(raw, ops, **options)["operators"]
               if row["operator"]["family"] == "attention")
    assert row["program"] == program
    memory = {key: value for key, value in row["declared_service"].items() if key.startswith("memory@")}
    assert memory and all(row["serviced"][key] == pytest.approx(value, rel=1e-12, abs=1e-3)
                          for key, value in memory.items())
    execution = dict(row["operator"]["metadata"])["program_execution"]
    if mode == "decode":
        assert next(iter(memory.values())) == pytest.approx(4_239_504)
        assert execution["kv_cache_access"]["storage"] == "paged" and execution["kv_cache_access"]["page_size"] == 128
        physical = execution["decode_physical_contract"]
        assert physical["conversion"]["scope"].startswith("on_chip") and physical["workspace"]["owner_slots"] > 0
        assert physical["workspace"]["capacity_bytes"]["total"] > 0
        assert {"q_l1", "scores", "k_l0b", "v_l0b"} <= {row["name"] for row in physical["buffer_lifecycle"]}

@pytest.mark.parametrize("homogeneous", [False, True])
def test_pipeline_tail_dependency_matches_the_literal_window(homogeneous):
    issued = []
    def tile(index):
        issued.append(index)
        return (Stage("a", {"r": 1}), Stage("b", {"b": 1}, after=("a",) if index == 64 else ()))
    engine = Engine([Resource(name, 1) for name in ("r", "b")])
    program = PipelineProgram(65, 1, tile, homogeneous=homogeneous)
    if not homogeneous:
        with pytest.raises(ModelingError, match="declared_homogeneity"):
            program.template()
    done = program.start_template(engine, "p", engine.submit("ready"))
    engine.advance(until=(done,))
    assert issued == list(range(65))
    assert done.time == 66 and engine.serviced == {"r": 65, "b": 65}

def test_phase_residue_and_periodic_keys_preserve_reference_geometry():
    groups = tuple((Stage("work", {"r": i}),) for i in (1, 2))
    phase = PipelinePhase("full", groups[0], 10, residue_stages=groups, end=42)
    assert [phase.stages_for(i) for i in (0, 1, 3)] == [groups[0], groups[1], groups[1]] and phase.end == 42
    template = PipelineProgram(4, 2, _tile, homogeneous=True).template(panels_per_output=2,
                                                                       key_of=lambda i: (i, i % 2))
    assert [(p.kind, p.key) for p in template.phases] == [(kind, (i, i % 2))
                                                        for i, kind in enumerate(("first", "last") * 2)]
    assert list(template.reference_stages()) == [_tile(i) for i in range(4)]

@pytest.mark.parametrize("floor", [0, 10])
def test_program_work_and_explicit_observation_floor_are_independent(floor):
    engine = _pipeline_engine()
    program = KernelProgram((Stage("work", {"r": 1}),), minimum_ms=floor)
    done = program.start(engine, "kernel", engine.submit("ready"))
    engine.advance(until=(done,))
    assert done.time == max(1, floor) and engine.serviced["r"] == 1

def test_phase_identity_preserves_typed_transport_and_zero_work_join():
    context = _context(8, 8, 8)
    join = replace(op("join", deps=("a", "b"), stream="aux"), family="barrier", recipe="join", dtype="", layout="")
    case = SimpleNamespace(model=None, machine=None, runtime=SimpleNamespace(vllm_impl="vllm", vllm_version="1", kernel_format=""))
    identity = service_identity(case, join, scope="step")
    assert identity["dtype"] == identity["layout"] == "" and identity["hardware"] == {}
    assert identity["deps"] == ["a", "b"] and identity["stream"] == "aux" and identity["replay_scope"] == "step"
    options = TransportOptions(hop_startup_us=2.5, receive_capacity_bytes=((1, 10), (0, 20)),
                               receive_buffer_ids=((1, "rx1"), (0, "rx0")), receive_buffer_id="rx",
                               max_inflight_per_rank=3, max_physical_chunks=77)
    context = replace(context, service_identity={"transport_options": options})
    derived = context.for_phase(Work(resources={"hbm": 1000}), kernel="copy", dtype="bf16", layout="contiguous")
    assert derived.transport_options is options and json.dumps(derived.service_identity)
    encoded = derived.transport_identity
    assert encoded["receive_capacity_bytes"] == [[0, 20], [1, 10]]
    assert encoded["receive_buffer_ids"] == [[0, "rx0"], [1, "rx1"]]
    assert (encoded["max_inflight_per_rank"], encoded["max_physical_chunks"], encoded["hop_startup_us"]) == (3, 77, 2.5)
