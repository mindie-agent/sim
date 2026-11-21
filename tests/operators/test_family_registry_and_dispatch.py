"""Governed family routing and physical execution contracts.

[Role] Exercise explicit family and source bindings, preparation and replay.
[Invariants] One owner per recipe and family; typed boundaries preserve work,
  dependencies, resource ownership and capture identity.
[Timeouts & Locks] Bounded simulation and retained fixture reads. No locks.
[Side-Effect Policy] No device work. Fixture mutations use pytest temporary paths.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
from itertools import product
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace
import pytest
from _support.axes import REPRESENTATIVE_SKUS, SKUS
from _support.hardware import hardware_case, isa_of, op as simple_op, sku_case
from mindie_sim.catalog.ops import gemm_op, vector_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.observations.historical import call_inventory, call_observations, compare_call
from mindie_sim.observations.kernels import kernel_rows
from mindie_sim.prediction.blocks import norm
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators
from mindie_sim.prediction.operators import (FamilyDispatcher, FamilyRegistry, OperatorPlanningContext,
                                             attention, mla_prolog, paged_mla, rotary, sparse_indexer)
from mindie_sim.prediction.program import KernelProgram, RecipeBook, RecipeContext, ResourceEnvelope, Stage, stage_from_fragment
from mindie_sim.prediction.results import FusedPipelineDescriptor, OpOccurrence
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import activation_quantization, lower_implementations, norm_quant, residual_norm
from mindie_sim.prediction.verification import verify_operator
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import ArchId
from mindie_sim.schema.invocation import Occurrence
from mindie_sim.schema.operators import AllocationEffect, FieldDisposition, IterationWork, ProgramFragment
from mindie_sim.schema.operators import ResourcePolicy, SemanticAttr, StateEffect, WorkItem, WorkSet
from mindie_sim.schema.subgraphs import EventPortSpec, PortSet, StatePortSpec, TensorPortSpec
from mindie_sim.schema.tensors import TensorLayout, TensorValue
from mindie_sim.schema.workload import IterationPlan, RequestView
from tools.run_operator_cases import control_case, run
pytestmark = pytest.mark.tier0
ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures"
CAPTURE = FIXTURES / "observations/operator-kernels.csv"
OPS = {
    rotary.register_family(): replace(simple_op("rotation", "rotary"), family="rotary", local_shape=(2, 8, 64)),
    mla_prolog.register_family(): mla_prolog.make_op(op_id="prolog", tokens=2, hidden_size=1024, num_heads=8, weight_quant_mode=0),
    paged_mla.register_family(): paged_mla.make_op(sequences=((1, 128),), logical_heads=16, physical_heads=16),
    attention.register_family(): attention.make_attention_op(
        attention.AttentionContract(1, 8, 1, 128, 128, sequences=((1, 128),)), op_id="attention"),
    sparse_indexer.register_family(): sparse_indexer.make_indexer_op(
        query_tokens=1, scanned_rows=128, index_heads=8, index_dim=128),
}
SPECS = tuple(OPS)
REGISTRY = FamilyRegistry(SPECS)
@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.name)
def test_explicit_recipe_and_implicit_family_have_one_owner(spec):
    op = OPS[spec]
    assert all(REGISTRY.lookup(replace(op, family="generic", recipe=r)) is spec for r in spec.owned_recipe_ids)
    assert all(REGISTRY.lookup(replace(op, family=f, recipe="")) is spec for f in spec.families)
@pytest.mark.parametrize("conflict,error", [
    ({}, "duplicate_operator_family_spec"),
    ({"name": "overlap", "owned_recipe_ids": frozenset()}, "duplicate_operator_family_owner"),
    ({"name": "overlap", "families": frozenset()}, "duplicate_operator_recipe_owner"),
])
def test_registry_rejects_each_kind_of_duplicate_ownership(conflict, error):
    with pytest.raises(ModelingError, match=error):
        FamilyRegistry((*SPECS, replace(SPECS[0], **conflict)))
@pytest.mark.parametrize("hardware", SKUS)
@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.name)
def test_dispatch_projects_prepared_work_from_declared_architecture(hardware, spec):
    op = OPS[spec]
    geom = Geometry.from_devices(sku_case(hardware).machine.dies)
    ctx = OperatorPlanningContext(OpOccurrence(0, 0, 0, op.op_id), geom, geom.datapath)
    result = FamilyDispatcher(REGISTRY).resolve_plan(op, ctx)
    plan, descriptor = result.recipe, result.descriptor
    assert plan.metadata["arch_id"] == isa_of(hardware)
    assert isinstance(descriptor, FusedPipelineDescriptor) and descriptor.owner == ctx.occurrence
    assert descriptor.member_op_ids == (op.op_id,) and descriptor.family == spec.name
    assert descriptor.phases == plan.pipeline_spec.phases and descriptor.sources == result.sources
    assert (result.op.hbm_bytes, dict(result.op.matrix_flops)) == (plan.work.resources.get("hbm", 0), plan.work.matrix)
    unknown = replace(ctx, datapath=replace(ctx.datapath, compute_arch=replace(ctx.compute_arch, arch_id=ArchId("unknown"))))
    if spec.name != "rotary":  # Rotary is route-based.
        with pytest.raises(UnmodeledCapabilityError):
            FamilyDispatcher(REGISTRY).resolve_plan(op, unknown)
def _iteration(op, *others, replay_scope="none"):
    plan = IterationPlan((RequestView("r", 1, 0, True, "prefill"),))
    return IterationWork((op, *others), plan, 1, 1, 1, None, "prefill", "source", replay_scope=replay_scope)
def _execute(case, book, iteration):
    execution = Execution(case, CostModel(case), recipes=book)
    done = execution.add(iteration, 0, 0).completion
    execution.engine.advance(until=(done,))
    return done.time, execution
def _products(op, geometry):
    loads, bf16, fp32 = Work(resources={"hbm": 1000}), Work(matrix={"bf16": 2000}), Work(matrix={"fp32": 2000})
    def program(ctx):
        return KernelProgram((
            Stage("load", ctx.bind(loads)), Stage("bf16", ctx.bind(bf16), after=("load",)),
            Stage("fp32", ctx.bind(fp32), after=("load",)),
            Stage("store", ctx.bind(loads), after=("bf16", "fp32"))))
    return OperatorPlan(loads + bf16 + fp32 + loads, program)
def test_mixed_precision_work_shares_one_pipe_and_survives_both_execution_paths():
    raw = hardware_case()
    op = replace(simple_op("mixed", "mixed"), requests=("r",), flops=999, hbm_bytes=999, vector_ops=999)
    book = RecipeBook({op.recipe: PlannedRecipe(_products)})
    duration, it = _execute(parse_case(raw), book, _iteration(op))
    row = operator_costs(raw, (op,), recipes=book)["operators"][0]
    assert duration == row["duration_ms"] == 5 and it.work["flops"] == row["operator"]["flops"] == 4000
    assert {k: v for k, v in it.engine.serviced.items() if v} == row["serviced"] == {"matrix@r0:s0": 3, "memory@r0:s0": 2000}
    raw["machine"]["device"]["compute_peaks"]["fp32"] = 2e6
    faster = verify_operator(raw, replace(op, layer=99), _products)["result"]
    assert faster["duration_ms"] == 4 and faster["serviced"]["memory@r0:s0"] == 2000
    del raw["machine"]["device"]["compute_peaks"]["fp32"]
    with pytest.raises(ValueError, match="compute_precision_rate_required:fp32"):
        verify_operator(raw, op, _products)
def _pair(op):
    return (replace(op, op_id="first", metadata=(("part", 1),)),
            replace(op, op_id="second", deps=("first",), metadata=(("part", 2),)))
def test_source_expansion_preserves_launches_and_downstream_dependencies():
    raw = hardware_case()
    raw["machine"]["device"]["launch_us"] = 50
    case, op = parse_case(raw), replace(simple_op("mixed", "mixed"), requests=("r",))
    consumer = replace(op, op_id="consumer", recipe="direct", deps=(op.op_id,))
    book = RecipeBook({op.recipe: PlannedRecipe(_products, _pair), "direct": PlannedRecipe(_products)})
    isolated = operator_costs(case, (op,), recipes=book)
    assert isolated["operator_occurrences"] == 2 and isolated["operator_sum_ms"] == 10
    for scope, launches in (("none", 3), ("full", 1)):
        duration, execution = _execute(case, book, _iteration(op, consumer, replay_scope=scope))
        assert duration == pytest.approx(15.05) and execution.work["flops"] == 12000
        assert execution.engine.serviced["host@r0:s0"] == pytest.approx(launches * .05)
def test_finite_work_audit_rejects_omissions_cancellation_and_wrong_dispatch():
    op, raw = simple_op("mixed", "mixed"), hardware_case()
    def missing(op, geom):
        plan = _products(op, geom)
        return replace(plan, work=plan.work + Work(resources={"hbm": 128}))
    row = verify_operator(raw, op, missing)
    assert row["finite_completion"] and row["status"] == "failed"
    assert row["work_mismatches"]["memory@r0:s0"] == {"declared": 2128, "executed": 2000}
    def offsets(op, geom):
        actual = Work(resources={"hbm": 2000 if dict(op.metadata)["part"] == 2 else 0})
        return OperatorPlan(Work(resources={"hbm": 1000}), lambda ctx: KernelProgram((Stage("copy", ctx.bind(actual)),)))
    row = verify_operator(raw, op, offsets, kernels=_pair)
    assert row["result"]["declared_service"] == row["result"]["serviced"]
    assert row["status"] == "failed" and len(row["work_mismatches"]) == 2
    module = SimpleNamespace(make_op=lambda params: op, build=_products)
    result = run(module, [dict(id="call", split="development", params={},
                               expected_dispatch=dict(state_v_first=True, use_exp2=True))])
    assert result["status"] == "failed"
    for row in result["cases"]:
        assert row["finite_completion"] and not row["work_mismatches"]
        assert row["dispatch_mismatches"] == dict.fromkeys(("state_v_first", "use_exp2"), {"expected": True, "selected": None})
def test_work_audit_detects_input_service_before_readiness():
    class EagerCopy:
        def start(self, engine, name, ready, *, owner=""):
            early = engine.submit(name + ":copy", {"memory@r0:s0": 1000})
            return engine.submit(name + ":join", after=(ready, early))
    def build(op, geom):
        return OperatorPlan(Work(resources={"hbm": 1000}), lambda ctx: EagerCopy())
    raw, op = hardware_case(), simple_op("source", "source")
    assert verify_operator(raw, op, build)["status"] == "passed"
    delayed = verify_operator(raw, op, build, input_ready_ms=2)
    assert delayed["finite_completion"] and delayed["status"] == "failed"
    assert delayed["result"]["service_before_input"] == {"memory@r0:s0": 1000}
@pytest.mark.parametrize("explicit", (False, True))
def test_copy_modes_share_a_pipe_and_keep_fixed_per_owner_capacity(explicit):
    op, raw = simple_op("copy", "copy"), hardware_case()
    device = raw["machine"]["device"]
    def build(op, geom):
        work = Work(resources={"copy": 1000, "strided": 1000})
        def program(ctx):
            return KernelProgram(tuple(Stage(mode, ctx.bind(Work(resources={mode: amount})),
                                             placement=(ctx.placement(mode, (0,)) if explicit
                                                        else ctx.owned(0, geom.compute_units, (mode,))))
                                       for mode, amount in work.resources.items()))
        return OperatorPlan(work, program)
    for units in (1, 4, 8):
        device["compute_units"], device["rates"]["copy"] = units, units * 1e6
        for rate, expected in ((units * .5e6, 3), (units * 1e6, 2)):
            device["service_modes"] = {"strided": {"resource": "copy", "rate": rate}}
            row = verify_operator(raw, op, build, input_ready_ms=7)
            assert row["status"] == "passed" and row["result"]["duration_ms"] == expected
    raw["model"].update(hidden_size=16, num_heads=2, num_kv_heads=2)
    raw["runtime"]["tp"] = raw["machine"]["num_devices"] = 2
    raw["machine"]["devices"] = [device, dict(device, compute_units=1)]
    with pytest.raises(ValueError, match="heterogeneous_owners_require_per_rank_execution"):
        verify_operator(raw, op, build)
def _fragment(name="move", amount=64, **options):
    return ProgramFragment(name, work=WorkSet((WorkItem("hbm", amount, "bytes"),)), **options)
def test_tensor_view_rebinds_without_losing_storage_or_precision():
    value = TensorValue("p", "bf16", TensorLayout.contiguous((2, 4)), quant="w8a8", layout_kind="nz",
                        storage="buf", alias_of="base", state_version=3, binding="event", binding_name="ready")
    view = value.narrow(1, 1, 2)
    assert view == replace(value, layout=TensorLayout((2, 2), (4, 1), 1, (2, 4)))
    assert view.with_binding("static") == replace(view, binding="static", binding_name="")
def test_fragment_serializes_typed_ports_scope_and_attrs():
    fragment = _fragment(occurrence=Occurrence("kernel", "call", request_ids=("r",), rank=3),
                         ports=PortSet(tensor_in=(TensorPortSpec("x", "in", "activation", dtype="bf16"),),
                                       state_out=(StatePortSpec("kv", "out", "paged_kv", scope="layer", unit="bytes"),),
                                       event_in=(EventPortSpec("ready", "in", "ready", generation=4),)),
                         attrs=(SemanticAttr("epsilon", 1e-6, source="declared"),))
    payload = fragment.to_dict()
    assert (payload["occurrence"]["rank"], payload["attrs"][0]["value"]) == (3, 1e-6)
    ports, expected = payload["ports"], ("bf16", "paged_kv", 4)
    assert (ports["tensor_in"][0]["dtype"], ports["state_out"][0]["representation"], ports["event_in"][0]["generation"]) == expected
@pytest.mark.parametrize("factory,error", [
    (lambda: WorkItem("hbm", -1, "bytes"), "work_item_amount_invalid"),
    (lambda: WorkSet((WorkItem("hbm", 1, "bytes"),) * 2), "work_set_duplicate"),
    (lambda: WorkSet((WorkItem("hbm", 1, "bytes"), WorkItem("hbm", 1, "ops", kind="useful"))), "work_set_unit_conflict"),
    (lambda: TensorValue("p", "bf16", TensorLayout.contiguous((1,)), binding="symbolic"), "binding_name_required"),
    (lambda: TensorValue("p", "bf16", TensorLayout.contiguous((1,)), binding="unknown"), "binding_unknown"),
    (lambda: PortSet(tensor_in=(TensorPortSpec("x", "in", "activation"),),
                     state_out=(StatePortSpec("x", "out", "cache"),)), "port_set_name_duplicate"),
    (lambda: EventPortSpec("x", "in", "guessed"), "event_port_kind_unknown"),
    (lambda: Occurrence("request", "r", request_ids=("r", "r")), "occurrence_request_ids_duplicate"),
    (lambda: Occurrence("request", "r", request_ids="r"), "scope_ids_require_sequence"),
    (lambda: StateEffect("kv", "commit"), "commit_version_required"),
    (lambda: FieldDisposition("activation", "rewrite"), "field_disposition_source_required"),
    (lambda: _fragment(state_effects=(StateEffect("kv", "commit", version=1),
                                     StateEffect("kv", "commit", version=2))), "state_effect_conflict_requires_arbiter"),
    (lambda: ResourcePolicy(sharing="fifo", preemptive=True), "resource_preemption_unsupported"),
    (lambda: ResourcePolicy(sharing="round_robin"), "resource_sharing_unknown"),
    (lambda: _fragment(policies={"unused": ResourcePolicy()}), "policy_resource_unused"),
    (lambda: _fragment(repeats=0), "repeats_invalid"),
])
def test_typed_contract_rejects_ambiguous_or_unexecutable_values(factory, error):
    with pytest.raises((ModelingError, ValueError, TypeError), match=error):
        factory()
@pytest.mark.parametrize("options,blocker", [
    ({"repeats": 2}, "repeats"), ({"attrs": (SemanticAttr("epsilon", 1e-6),)}, "attrs"),
    ({"policies": {"hbm": ResourcePolicy(sharing="fifo")}}, "policies"),
    ({"state_effects": (StateEffect("cache", "read"),)}, "state_effects"),
    ({"allocations": (AllocationEffect.acquire("buf", "l2", 64, condition="if ready"),)}, "conditional_allocation"),
    ({"allocations": (AllocationEffect.acquire("buf", "l2", 64, lifetime="program"),)}, "allocation_lifetime"),
    ({"ports": PortSet(event_in=(EventPortSpec("ready", "in", "ready"),))}, "event_in"),
    ({"ports": PortSet(event_out=(EventPortSpec("commit", "out", "commit"),))}, "event_out"),
    ({"ports": PortSet(event_out=tuple(EventPortSpec(name, "out", "complete") for name in ("a", "b")))}, "event_out"),
])
def test_static_stage_projection_rejects_fragment_semantics_it_cannot_execute(options, blocker):
    with pytest.raises(ModelingError, match=f"stage_from_fragment_unsupported:{blocker}"):
        stage_from_fragment(_fragment(**options))
@pytest.mark.parametrize("sharing,order,priority,expected", [
    ("fifo", ("a", "b"), 0, (1, 3, 4)), ("fair", ("a", "b"), 0, (2, 3, 5)),
    ("fifo", ("b", "a"), 0, (3, 2, 6)), ("fifo", ("b", "a"), 1, (1, 3, 4)),
])
def test_fragment_projection_conserves_work_under_sharing_and_priority(sharing, order, priority, expected):
    rule = ResourcePolicy(sharing=sharing).engine_kwargs()
    engine = Engine([Resource("hbm", 64, unit="bytes", **rule), Resource("vector", 1, unit="ops")])
    ready = engine.submit("ready")
    fragments = {"a": _fragment("a"), "b": _fragment("b", 128, priority=priority)}
    done = {name: KernelProgram((stage_from_fragment(fragments[name]),)).start(engine, name, ready)
            for name in order}
    tail = ProgramFragment("tail", work=WorkSet((WorkItem("vector", 3, "ops"),)))
    done["tail"] = KernelProgram((stage_from_fragment(tail),)).start(engine, "tail", done["a"])
    engine.advance(until=tuple(done.values()), first=False)
    assert tuple(done[name].time for name in ("a", "b", "tail")) == expected
    assert engine.serviced == {"hbm": 192, "vector": 3}
def test_projection_charges_physical_work_and_retains_allocations_until_completion():
    engine = Engine([Resource("hbm", 64)], pools={"l2": 64}, trace_limit=20)
    alloc = (AllocationEffect.acquire("buf", "l2", 64), AllocationEffect.release("buf", "l2", 64))
    work = WorkSet((WorkItem("hbm", 64, "bytes"), WorkItem("hbm", 4096, "bytes", kind="useful")))
    fragment = ProgramFragment("move", allocations=alloc, work=work,
                               ports=PortSet(event_out=(EventPortSpec("done", "out", "complete"),)))
    first, second = (stage_from_fragment(fragment, latency=latency) for latency in (5, 0))
    assert first.fragment is fragment and first.work == {"hbm": 64} and first.work_units == {"hbm": "bytes"}
    a, b = (KernelProgram((stage,)).start(engine, name, engine.submit(name, release=i))
            for i, (stage, name) in enumerate(((first, "a"), (second, "b"))))
    engine.advance(until=(a, b), first=False)
    assert (a.time, b.time) == (5, 6) and engine.serviced["hbm"] == 128 and not engine.allocations
def test_coupling_retains_both_resources_and_ties_follow_submission_order():
    for coupled in (True, False):
        engine = Engine([Resource("a", 10), Resource("b", 10)], trace_limit=20)
        first, second = engine.submit("first"), engine.submit("second")
        ready = engine.submit("join", after=(second, first))
        fragment = ProgramFragment("work", work=WorkSet((WorkItem("a", 30, "ops"), WorkItem("b", 10, "ops"))),
                                   coupled=coupled)
        done = KernelProgram((stage_from_fragment(fragment),)).start(engine, "run", ready)
        engine.advance(until=(done,), first=False)
        assert [row["task"] for row in list(engine.trace)[:3]] == ["first", "second", "join"]
        assert engine.serviced == {"a": 30, "b": 10} and done.time == 3
        assert engine.busy_ms == {"a": 3, "b": 3 if coupled else 1}
EXTENSIONS = ("cuda_norm", "cuda_quant", "attention", "grouped_gemm", "moe_align",
              "moe_combine", "gdn_prepare", "gdn_state")
def _source_case(name):
    if name == "attention":
        return attention, replace(OPS[attention.register_family()], requests=("r",)), sku_case(REPRESENTATIVE_SKUS["a5"])
    module = importlib.import_module("mindie_sim.prediction.operators." + name)
    params = json.loads((FIXTURES / "operator_cases" / f"{name}.json").read_text())["cases"][0]["params"]
    op = replace(module.make_op(params), requests=("r",))
    return module, op, parse_case(control_case(units=4, service_roles=getattr(module, "SERVICE_ROLES", {})))
@pytest.mark.parametrize("name", [name for name in EXTENSIONS if name != "attention"])
def test_source_calls_require_explicit_binding_in_both_execution_paths(name):
    module, op, case = _source_case(name)
    bound, op = PlannedRecipe(module.build), replace(op, metadata=(*op.metadata, ("program_kind", "matmul")))
    for book in (RecipeBook(), RecipeBook(families={op.family: bound}),
                 RecipeBook(programs={"matmul": bound}), RecipeBook({op.recipe: ResourceEnvelope()})):
        with pytest.raises(ModelingError, match="source_plan_binding_required"):
            operator_costs(case, (op,), recipes=book)
        execution = Execution(case, CostModel(case), recipes=book)
        with pytest.raises(ModelingError, match="source_plan_binding_required"):
            execution.add(_iteration(op), 0, 0)
        assert not any(execution.engine.serviced.values())
def test_resource_envelope_cannot_execute_even_when_bound_to_a_recipe():
    op = replace(simple_op("unmodeled", "envelope"), hbm_bytes=128)
    with pytest.raises(ModelingError, match="resource_envelope_banned"):
        operator_costs(hardware_case(), (op,), recipes=RecipeBook({op.recipe: ResourceEnvelope()}))
@pytest.mark.parametrize("name", (*EXTENSIONS[:3], "moe_align"))
def test_bound_source_work_matches_iteration_and_expansion_is_idempotent(name):
    module, op, case = _source_case(name)
    expand = getattr(module, "kernels", None)
    book = RecipeBook({op.recipe: PlannedRecipe(module.build, expand)})
    isolated = operator_costs(case, (op,), recipes=book)
    duration, execution = _execute(case, book, _iteration(op))
    assert duration == pytest.approx(isolated["operator_sum_ms"]) and duration > 0
    expected = Counter()
    for row in isolated["operators"]:
        assert row["program"] == "OperatorPlan"
        expected.update({k: v * row["occurrences"] for k, v in row["serviced"].items() if v})
    assert {k: v for k, v in execution.engine.serviced.items() if v} == pytest.approx(expected)
    if op.source_boundary == "call":
        unexpanded = RecipeBook({op.recipe: PlannedRecipe(module.build)})
        for execute in (lambda: operator_costs(case, (op,), recipes=unexpanded),
                        lambda: _execute(case, unexpanded, _iteration(op))):
            with pytest.raises(ModelingError, match="source_kernel_expansion_required"):
                execute()
        with pytest.raises(ModelingError, match="source_call_requires_kernel_expansion"):
            book.prepare(op, Geometry(4))
        children = book.expand((op,))
        assert len(children) == len(expand(op)) + 1 and book.expand(children) == children
        assert all(operator_costs(case, (child,), recipes=book)["operator_occurrences"] == 1 for child in expand(op))
def test_semantic_fusion_cannot_rewrite_source_owned_operands():
    module, op, case = _source_case("cuda_norm")
    book = RecipeBook({op.recipe: PlannedRecipe(module.build)})
    payload = dict(stage=0, layer=0, requests=("r",), elements=5120, dtype="bf16", phase="prefill")
    quant = vector_op(op_id="quant", deps=(op.op_id,), ops_per_element=4, family="quantize", recipe="dynamic_quantize", **payload)
    add = vector_op(op_id="add", deps=(), ops_per_element=1, family="elementwise", recipe="residual_add", **payload)
    for before, transform in (((op, quant), norm_quant), ((add, replace(op, deps=("add",))), residual_norm)):
        assert operator_costs(case, transform(before, "bf16"), recipes=book) == operator_costs(case, before, recipes=book)
    source = replace(_matrix("source", (), quant=quant_preset("w8a8_int8")), source_boundary="kernel")
    assert activation_quantization((source,), "bf16") == [source]
def _historical():
    root = FIXTURES / "observations"
    return [json.loads((root / name).read_text()) for name in ("gdn-call-protocol.json", "gdn-call-measure.json")]
def test_capture_keeps_trials_canaries_and_call_comparison_boundaries():
    protocol, raw = _historical()
    rows = list(call_observations(protocol, root=ROOT))
    for source, observed in zip(raw["rows"], rows, strict=True):
        samples = source["samples_ms"]
        assert [s["duration_ms"] for s in observed["samples"]] == samples
        assert [s["trial"] for s in observed["samples"]] == [0] * 20 + [1] * 20
        assert observed["statistics"]["sample_median_ms"] == median(samples)
        assert observed["statistics"]["median_of_trial_medians_ms"] == median((median(samples[:20]), median(samples[20:])))
        assert observed["reported_statistics"]["median"] == source["stat_median"]
    by_role = {r["identity"]["conditions"]["role"]: r for r in rows}
    assert by_role["scan_prefill"]["canary"]["ok"] is False
    observed = by_role["scan_decode"]
    base = dict(observed, duration_ms=1.1 * observed["statistics"]["median_of_trial_medians_ms"])
    assert compare_call(base, observed)["comparison"]["relative_error"] == pytest.approx(.1)
    geometry, components = deepcopy(base), deepcopy(base)
    geometry["identity"]["conditions"]["batch"] = 4
    components["identity"]["call_contract"]["components"].pop(0)
    assert all(compare_call(row, observed)["comparison"] is None for row in (geometry, components, dict(base, boundary="device_kernel")))
    inv = call_inventory(protocol, root=ROOT, preview_limit=2)
    assert (inv["condition_rows"], len(inv["observation_preview"])) == (len(rows), 2)
    assert inv["event_samples"] == sum(len(r["samples_ms"]) for r in raw["rows"])
def test_capture_changes_require_new_identity_and_failed_rows_have_no_service(tmp_path):
    protocol, raw = _historical()
    path = tmp_path / "capture.json"
    failure = {k: v for k, v in raw["rows"][0].items() if not k.startswith("stat_") and k != "samples_ms"}
    raw["rows"] = [dict(failure, status="failed", error_first_line="warm-up failure")]
    path.write_text(json.dumps(raw))
    protocol["capture"]["path"] = str(path)
    with pytest.raises(ValueError, match="historical_source_changed"):
        list(call_observations(protocol, root=ROOT))
    protocol["capture"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    row = next(call_observations(protocol, root=ROOT))
    assert row["statistics"] is None and not row["samples"] and row["failure"]["error_first_line"] == "warm-up failure"
    assert compare_call(dict(row, duration_ms=0), row)["comparison"] is None
def _matrix(name, deps, m=2, n=16, k=8, quant=None):
    return gemm_op(op_id=name, deps=deps, stage=0, layer=0, requests=("r",), m=m, n=n, k=k,
                   quant=quant or quant_preset("mxfp8"), category="dense_mlp", phase="prefill")
def test_quantized_input_is_shared_and_activation_grouping_is_independent():
    case = parse_case(hardware_case())
    runtime = replace(case.runtime, lowering_passes=("activation_quantization", "norm_quant", "residual_norm"))
    q, a, b = lower_implementations([_matrix("a", ("input",)), _matrix("b", ("input",), n=32)], runtime, "bf16")
    assert a.deps == b.deps == (q.op_id,)
    assert (q.mte2_bytes, q.mte3_bytes) == (2 * 8 * 2, 2 * 8 + 2 * 2)
    assert a.mte2_bytes == q.mte3_bytes + 16 * 8 + 16
    integer = _matrix("integer", ("input",), quant=quant_preset("w8a8_int8"))
    q, integer = lower_implementations([integer], runtime, "bf16")
    assert q.mte3_bytes == 2 * 8 + 2 * 4 and integer.dtype == "int8"
    spec = replace(quant_preset("mxfp8"), weight_block=(128, 128), scale_format="fp32", scale_bytes=4,
                   activation_group_size=64, activation_scale_format="fp32", activation_scale_pack=1)
    op = _matrix("blocked", (), n=128, k=129, quant=spec)
    assert (dict(op.metadata)["activation_scale_shape"], op.mte2_bytes) == ((2, 3), 128 * 129 + 2 * 4 + 2 * 129 + 2 * 3 * 4)
def test_tensor_reshape_retains_element_addresses_and_requires_real_copies():
    layout = TensorLayout.contiguous((3, 8, 4))
    sliced = layout.narrow(1, 2, 4)
    for current, shape in ((layout, (96,)), (sliced, (3, 4, 2, 2)),
                           (layout.permute((1, 0, 2)), (8, 3, 4)), (sliced.narrow(0, 0, 1), (16,))):
        view = current.reshape_view(shape)
        assert view is not None
        def addresses(tensor):
            return [tensor.offset + sum(i * s for i, s in zip(index, tensor.strides))
                    for index in product(*(range(extent) for extent in tensor.shape))]
        assert addresses(view) == addresses(current)
    assert sliced.reshape_view((12, 4)) is None
def test_retained_native_gdn_and_convolution_rows_match_active_frontend_contracts(profiler_api):
    from tools.compare_operators import GDN_KINDS, observed_contract, predicted_contract
    rows = {row.kind: row for row in kernel_rows(FIXTURES / "observations/gdn-native-kernels.csv")}
    raw = json.loads((ROOT / "examples/cases/qwen38-a5.json").read_text())
    ops = {op.family: op for op in target_operators(raw, phase="prefill", tokens=16384)}
    assert all(predicted_contract(ops[f].to_dict()) == observed_contract(rows[k]) for f, k in GDN_KINDS.items())
    gate = rows["ChunkGatedDeltaRuleFwdPrepare"]
    assert observed_contract(replace(gate, input_dtypes=("DT_BF16",) * 5 + ("INT64",) * 2)) != observed_contract(gate)
    conv = ops["causal_conv1d_fn"]
    capture = next(kernel_rows(FIXTURES / "observations/native-causal-conv-kernel.csv"))
    meta = dict(conv.metadata)
    assert predicted_contract(conv.to_dict()) == observed_contract(capture)
    assert meta["state_rows"] != capture.input_shapes[3][0] and meta["state_history_slots"] == capture.input_shapes[3][1]
def test_capture_preserves_epoch_precision_unknown_shapes_and_strided_projection(profiler_api):
    rows = list(kernel_rows(CAPTURE))
    matrix = next(row for row in rows if row.input_shapes == ((16384, 5120), (16384, 5120)))
    assert (matrix.start_ns, matrix.duration_ns) == (1792684744878053232, 6381070)
    assert any(row.kind == "fused_gdn_gating_kernel" and row.input_shapes is None and row.duration_ns > 0 for row in rows)
    raw = next(row for row in rows if row.kind == "Slice" and row.input_shapes[0] == (16384, 96))
    view = TensorLayout.contiguous(raw.input_shapes[0]).narrow(1, 48, 48)
    assert view.shape == raw.output_shapes[0] and not view.is_contiguous and view.elements * 2 == 1572864
    recurrent = next(row for row in rows if row.kind == "RecurrentGatedDeltaRule")
    assert (recurrent.core, recurrent.counters_us["aic_mac_time(us)"]) == ("AI_VECTOR_CORE", 0)
@pytest.mark.parametrize("case,capture,tokens,phase,recipe,kind", [
    ("glm53-a3-w4a8", "a3_glm53_prefill", 4096, "prefill", "index_norm", "LayerNormV3"),
    ("glm53-a3-w4a8", "a3_glm53_decode", 1, "decode", "v_expand", "TransposeBatchMatMul"),
    ("kimi-k3-a5", "a5_kimi_k3", 1, "decode", "v_expand", "TransposeBatchMatMul"),
    ("v4pro-a3-w4a8-tp16", "a3_v4pro", 1024, "prefill", "grouped_output_a", "TransposeBatchMatMul"),
])
def test_native_layout_matches_retained_geometry_precision_and_payload(case, capture, tokens, phase, recipe, kind, profiler_api):
    import mindie_profiler
    from mindie_profiler.body_layout_contracts import observed_layout_contract
    from mindie_sim.structure_contracts import predicted_contract
    raw = json.loads((ROOT / "examples/cases" / f"{case}.json").read_text())
    args = dict(phase=phase, tokens=tokens, context=16 if phase == "decode" else 0)
    def lower():
        ops = target_operators(raw, **args)
        return next(op for op in ops if op.recipe == recipe), ops
    if recipe == "v_expand" and case.startswith("glm53"):
        assert predicted_contract(lower()[0].to_dict()) is None
        raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})["mla_prolog"] = "vllm_ascend_mla_prolog_v3"
    path = Path(mindie_profiler.__file__).resolve().parents[1] / "tests/fixtures/fusion_contracts" / f"{capture}.csv"
    observed = observed_layout_contract(next(row for row in kernel_rows(path) if row.kind == kind))
    op, ops = lower()
    assert predicted_contract(op.to_dict()) == observed
    if recipe == "index_norm":
        assert (op.mte2_bytes, op.mte3_bytes) == ((tokens + 2) * 128 * 4, tokens * 128 * 4)
        assert predicted_contract(replace(op, dtype="bf16").to_dict()) != observed
        return
    h, m, n, k = observed[2]
    assert op.flops == 2 * h * m * n * k
    if recipe == "v_expand":
        assert (op.mte2_bytes, op.mte3_bytes) == (2 * h * (m * k + k * n), 2 * h * m * n)
        assert not any(node.recipe.startswith("v_expand_transpose_copy") for node in ops)
    else:
        raw["model"]["quant"]["category_overrides"]["mla.o_a"] = dict(weight_format="int8", activation_format="int8")
        quant, _ = lower()
        assert quant.flops == op.flops and predicted_contract(quant.to_dict()) is None
def test_fusion_preserves_residual_and_unquantized_consumers(profiler_api):
    case = parse_case(hardware_case())
    runtime = replace(case.runtime, lowering_passes=("activation_quantization", "norm_quant", "residual_norm"))
    residual = vector_op(op_id="residual", deps=("x", "saved"), stage=0, layer=0, requests=("r",),
                         elements=16, ops_per_element=1, dtype="bf16", family="elementwise",
                         phase="prefill", recipe="residual_add", read_bytes_override=64, write_bytes_override=32)
    def normalized(tokens, width, source):
        return norm(name="norm", deps=(source,), stage=0, layer=0, requests=("r",),
                    tokens=tokens, width=width, model=case.model, mode="prefill")
    n = normalized(2, 8, "residual")
    consumer = replace(residual, op_id="consumer", deps=("projection", "residual"))
    fused, mm, later = lower_implementations([residual, n, _matrix("projection", ("norm",)), consumer], runtime, "bf16")
    assert (fused.mte2_bytes, fused.mte3_bytes) == (64 + 8 * 2, 32 + 16 + 4)
    assert later.deps == (mm.op_id, fused.op_id) and fused.returns == ("activation_scales", "residual")
    direct = replace(consumer, deps=("norm",))
    n = normalized(2, 8, "input")
    ops = lower_implementations([n, _matrix("projection", ("norm",)), direct], runtime, "bf16")
    assert ops[0] == n and ops[-1].deps == ("norm",) and sum(op.family == "quantize" for op in ops) == 1
    captured = next(row for row in kernel_rows(CAPTURE) if row.kind == "AddRmsNormDynamicMxQuant")
    tokens, width = captured.input_shapes[0]
    large = replace(residual, mte2_bytes=tokens * width * 4, mte3_bytes=tokens * width * 2,
                    local_shape=(tokens * width,))
    fused = lower_implementations([large, normalized(tokens, width, "residual"),
                                   _matrix("projection", ("norm",), m=tokens, n=34816, k=width)], runtime, "bf16")[0]
    assert dict(fused.metadata)["activation_scale_shape"] == captured.output_shapes[2]
    assert fused.mte3_bytes == tokens * width * 3 + tokens * (width // 64) * 2
def test_vector_recurrence_keeps_ragged_state_and_distinct_gate_payloads():
    raw = hardware_case()
    raw["machine"]["device"]["resource_units"] = {"vector": 3}
    case = parse_case(raw)
    layer = replace(case.model.layer_plans[0], attention="gdn", linear_key_heads=1, linear_value_heads=2,
                    linear_heads=2, linear_key_head_dim=3, linear_head_dim=2, conv_kernel=3, recurrent_state_dtype="fp32")
    model = replace(case.model, layer_plans=(layer,))
    views = (RequestView("a", 2, 16, False, "decode"), RequestView("b", 4, 32, False, "decode"))
    ops, _ = Frontend().body(model, case.runtime, views, mode="decode")
    scan, = [op for op in ops if op.family == "recurrent_vector"]
    assert dict(scan.metadata)["sequence_tokens"] == (2, 4)
    assert (scan.flops, scan.mte2_bytes, scan.mte3_bytes) == (0, 4 * 24 + 12 * ((2 * 3 + 2) * 2 + 6), 12 * (24 + 2 * 2))
    serviced = operator_costs(case, (scan,))["operators"][0]["serviced"]
    assert serviced["vector@r0:s0"] == 7 * 12 * 3 * 2 and serviced["memory@r0:s0"] == scan.hbm_bytes
    by_recipe = {op.recipe: op for op in ops}
    gate = by_recipe["output_gate_reshape"]
    assert gate.deps == (by_recipe["linear_qkvz"].op_id,)
    assert gate.op_id in by_recipe["linear_output_norm_gate"].deps
    assert all(op.op_id not in by_recipe["linear_output_norm_gate"].deps for op in ops if op.family == "gdn_gating")
    runtime = replace(case.runtime, tensor_layouts={"gdn_gating": dict(beta="contiguous", alpha="contiguous"),
                                                   "short_convolution": {"input": "contiguous"}})
    for n in (1, 3):
        ops, _ = Frontend().body(model, runtime, (RequestView("r", n, 0, False, "prefill"),))
        copies = {op.recipe: op for op in ops if op.family == "tensor_copy"}
        assert bool(copies) == (n > 1)
        if copies:
            assert copies["gate_alpha_contiguous"].mte2_bytes == copies["gate_beta_contiguous"].mte3_bytes == n * 4
            assert copies["convolution_input_contiguous"].mte2_bytes == n * (2 * 3 + 2 * 2) * 2
def test_final_normalization_obeys_selected_row_scope_and_partial_prefill():
    case = parse_case(hardware_case())
    views = (RequestView("prompt", 8, 0, True, "prefill"), RequestView("verify", 3, 16, True, "decode"))
    for scope, sizes in (("all", (11, 4, 4)), ("selected", (4, 4, 4))):
        runtime = replace(case.runtime, final_norm_scope=scope)
        ops, _ = Frontend().logits(case.model, runtime, views, "body", mode="mixed")
        assert tuple(op.local_shape[0] for op in ops[:3]) == sizes
        assert ops[1].deps == (ops[0].op_id,) and ops[2].deps == (ops[1].op_id,)
    partial, _ = Frontend().logits(case.model, case.runtime, (replace(views[0], samples=False),), "body", mode="prefill")
    assert len(partial) == 1 and partial[0].local_shape[0] == 8
def test_tail_copy_supply_tracks_active_owners_and_scalar_store_masks_rows():
    from mindie_sim.prediction.matrix import TiledMatmul
    from mindie_sim.prediction.pointwise import TiledPointwise
    op = _matrix("tile", (), m=1, n=4, k=2)
    rates, times = dict(cube=10000, hbm=100000, mte2=100, mte3=100), []
    for distributed in (False, True):
        engine = Engine([Resource(k, v, units=4) for k, v in rates.items()])
        ctx = RecipeContext(op, {"cube": op.flops}, dict(zip(rates, rates)), rates, 4)
        event = TiledMatmul(1, 2, 2, distributed_transfers=distributed).lower(ctx).start(engine, "matrix", engine.submit("in"))
        engine.advance(until=(event,))
        times.append(event.time)
        assert engine.serviced["hbm"] == op.hbm_bytes
    assert times[1] - times[0] == pytest.approx((2 * op.mte2_bytes - dict(op.metadata)["packed_weight_bytes"] + op.mte3_bytes) / 100)
    roles = dict(hbm=1e20, mte2=1e20, mte3=1e20, vector=1e20, scalar_store=64000)
    recipe = TiledPointwise(4, ({"bytes_per_element": 4},
                                {"bytes_per_element": 2, "resource": "scalar_store", "bytes_per_work": 2}),
                            parameter_bytes_per_col=8)
    for rows, cols in ((1, 7), (64, 48), (4096, 48), (4225, 33), (16384, 48)):
        probe = replace(simple_op("point", "point"), local_shape=(rows, cols), vector_ops=12 * rows * cols)
        engine = Engine([Resource(k, v) for k, v in roles.items()])
        ctx = RecipeContext(probe, {}, dict(zip(roles, roles)), roles, 32, resource_units=dict.fromkeys(roles, 64))
        event = recipe.lower(ctx).start(engine, "point", engine.submit("in"))
        engine.advance(until=(event,))
        work, iterations = engine.serviced, (rows + 4095) // 4096
        assert event.time == pytest.approx(min(rows, iterations * 64) * cols / 1000)
        assert (work["scalar_store"], work["mte3"]) == (rows * cols, rows * cols * 4)
        assert work["mte2"] == rows * cols * 4 + 64 * iterations * cols * 8
        assert work["hbm"] == work["mte2"] + rows * cols * 6
        assert work["vector"] == 64 * iterations * ((cols + 7) // 8) * 64 * 8 * 12
