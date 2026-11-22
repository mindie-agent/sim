"""Service regimes: boundaries, continuity, physical rates, transport, geometry, knots.
[Role]
  Pin the observable-service-regime contract at its physical boundary: a native
  response is selected only inside its declared domain, continuity groups are
  gap-free and seam-matching, changed physical supply restores the analytical
  program, and unseen coordinates interpolate through the real Engine.
[Invariants]
  * Rule order never changes selection; interior gaps/overlaps and seam
    mismatches raise the exact declared ``ModelingError`` code.
  * A regime replaces only the priced response: work, serviced bytes and the
    single physical Engine stay identical to the analytical path.
  * Unseen points interpolate with no per-point installation; identity and
    domain misses fall back to the analytic program.
[Timeouts & Locks]
  Tier 0: bounded in-memory graphs; no device, network, sleep or file I/O.
[Side-Effect Policy]
  Local raw cases and ledgers; writes only pytest temporary state.
"""
from __future__ import annotations
from copy import deepcopy
from dataclasses import replace
from itertools import product
import json
import pytest
from mindie_sim import compose_case, graph_costs, operator_costs, predict
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction.body_defaults import attention_defaults
from mindie_sim.prediction.operator_analysis import target_graphs
from mindie_sim.prediction.efficiency.cost import primitive_key
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program.primitives import PrimitiveStore
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from mindie_sim.prediction.response_functions import (
    ResponseFunction, nonnegative, positive, scaled_coordinate)
from mindie_sim.prediction.service_regime_domains.services import ResourceService
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork
from _support.collective_cases import cohort_case, exchange, joint_case
from _support.hardware import hardware_case
from _support.kernel_invocation import (
    competitor, copy_op, curve as service_curve, fork, group, lower_copy, op, phase,
    raw_case, recipes)
from _support.serving_cases import (
    affine, bind, by_prefix, case, clean, compiled, curve, curve_case, declared_rule, engine,
    geometry_case, install, lowered, operation, rect, rejects, response, response_rule,
    roofline, rule, rule_for, select, single_device_machine, source_call,
)
pytestmark = pytest.mark.tier0

@pytest.mark.parametrize("response_us,complete", [(9000, 26), (15000, 29)])
def test_json_shape_axis_matches_immutable_source_and_inclusive_hardware_minimum(
        response_us, complete):
    raw = hardware_case()
    raw["machine"]["device"].update(launch_us=3000, kernel_invocations={
        "vector": {"startup_ms": 2, "minimum_ms": 12}})
    operation = replace(copy_op(4000), metadata=(
        ("invocation_class", "vector"), ("geometry", {"tiles": ((1, 2), (3, 4))})))
    install(raw, json.loads(json.dumps(response(response_us, match={
        "local_shape": [2000], "metadata.geometry": {"tiles": [[1, 2], [3, 4]]}}))))
    result = graph_costs(raw, (operation,), input_ready_ms=11, include_submission=True,
                         isolated=False, trace_limit=100)
    assert result["complete_ms"] == complete
    assert result["parameters"]["service_regimes"][0]["queries"] == {"observable_response": 1}
    clean(result)

@pytest.mark.parametrize("scope,hosts,host_ms", [("none", 2, 4), ("full", 1, 3)])
def test_two_native_groups_keep_one_response_and_one_host_submission_boundary(
        scope, hosts, host_ms):
    raw = install(hardware_case(), response(10000))
    raw["machine"]["device"].update(launch_us=2000, kernel_invocations={
        "vector": {"startup_ms": 3, "minimum_ms": 8}})
    raw["runtime"]["submission"] = {"graph_replay_ms": 3}
    result = graph_costs(raw, fork("first", ("late",)) + fork("second", ("first.exit",)),
                         replay_scope=scope, include_submission=True,
                         external_inputs={"late": 11}, isolated=False, trace_limit=300)
    submission = result["execution"]["submission"]
    assert submission["eager_invocations" if scope == "none" else "graph_replays"] == hosts
    assert submission["host_service_ms"] == host_ms
    assert result["execution"]["serviced"]["memory@r0:s0"] == 10000
    assert result["execution"]["serviced"]["vector@r0:s0"] == 6000
    assert result["complete_ms"] == 31
    assert result["parameters"]["service_regimes"][0]["queries"] == {"observable_response": 2}
    clean(result)

def test_regime_waits_for_finite_group_buffer_admission_and_is_not_added_to_startup():
    raw = install(raw_case(startup=2, minimum=8), response(12000))
    prior = replace(competitor(0, deps=()), op_id="prior", vector_ops=5000)
    operations = list(group())
    operations[0] = replace(operations[0], deps=("begin",))
    result = graph_costs(raw, (prior, *operations), recipes=recipes(),
                         external_inputs={"begin": 1}, isolated=False, trace_limit=200)
    admission = next(e for e in result["events"] if e["task"].endswith(":buffers_acquire"))
    assert admission["start_ms"] == 5
    assert result["complete_ms"] == 17
    clean(result)

@pytest.mark.parametrize("bandwidth,complete", [(.5, 20), (1, 19), (2, 19)])
def test_cross_dp_native_group_keeps_late_endpoint_wire_once_and_physical_bound(
        bandwidth, complete):
    raw = install(cohort_case(2), response(10000))
    raw["machine"]["links"][0]["capacity_bytes_per_ms"] = bandwidth
    raw["machine"]["device"].update(launch_us=2000, kernel_invocations={
        "vector": {"startup_ms": 3, "minimum_ms": 8}})
    raw["runtime"]["kernels"]["families"]["alltoall"] = {
        "kind": "direct_peer_collective", "max_inflight_per_rank": 1}
    graphs = {}
    for rank, volume in [(0, 0), (1, 4)]:
        entry = replace(phase("entry", entry=True, deps=("input" + str(rank),)),
                        vector_ops=0, metadata=phase("entry", entry=True).metadata
                        + (("invocation_class", "vector"),))
        transfer = replace(exchange(volume, axis="ep"), op_id="wire", deps=("entry",),
                           metadata=exchange(volume, axis="ep").metadata
                           + phase("wire").metadata)
        graphs[rank] = (entry, transfer, phase("exit", exit=True, deps=("wire",)))
    result = graph_costs(raw, graphs, external_inputs={"input0": 1, "input1": 9},
                         include_submission=True, isolated=False, trace_limit=300)
    assert result["complete_ms"] == complete
    assert sum(v for k, v in result["execution"]["serviced"].items() if k.startswith("link:")) == 4
    clean(result)

def test_existing_primitive_minimum_and_regime_are_inclusive_competing_constraints():
    raw = install(hardware_case(), response(12000, match={"recipe": copy_op(4000).recipe}))
    raw["runtime"]["kernels"]["invocations"] = [{
        "match": {"recipe": copy_op(4000).recipe}, "startup_ms": 3, "minimum_ms": 8}]
    operation = copy_op(4000)
    with PrimitiveStore(":memory:") as store:
        store.record(primitive_key(parse_case(raw), operation, "none"), duration_ms=20,
                     run_id="independent-control", repeat_index=0, source="test-known-bound")
        result = graph_costs(raw, (operation,), primitives=store, input_ready_ms=7,
                             isolated=False)
    assert result["complete_ms"] == 27
    assert result["execution"]["serviced"]["memory@r0:s0"] == 4000
    clean(result)

def test_group_link_scaling_uses_real_route_capacity_and_keeps_startup_on_the_critical_path():
    rule = response(0, coordinates={"bytes": {"path": "work.link"}}, response={"terms": [{
        "coordinate": "bytes", "us_per_unit": 2000,
        "scale_with": {"resource": "link", "reference_rate": 1000}}]})
    spans = []
    for factor in (.5, 1, 2):
        raw = install(cohort_case(2), rule)
        raw["machine"]["links"][0]["capacity_bytes_per_ms"] = factor
        raw["runtime"]["kernels"]["families"]["alltoall"] = {
            "kind": "direct_peer_collective", "max_inflight_per_rank": 1}
        raw["runtime"]["kernels"]["invocations"] = [{
            "match": {"kernel": "NativeDispatch"}, "startup_ms": 3}]
        graphs = {}
        for rank in range(2):
            move = replace(exchange(4, axis="ep"), op_id="wire", deps=("entry",),
                           metadata=exchange(4, axis="ep").metadata + phase("wire").metadata)
            graphs[rank] = (phase("entry", entry=True), move,
                            phase("exit", exit=True, deps=("wire",)))
        result = graph_costs(raw, graphs, isolated=False)
        spans.append(result["duration_ms"])
        assert sum(v for k, v in result["execution"]["serviced"].items() if k.startswith("link:")) == 8
        assert result["parameters"]["service_regimes"][0]["queries"] == {"observable_response": 2}
        clean(result)

def test_inclusive_resource_curve_keeps_real_competing_supply_under_outer_response():
    raw = install(hardware_case(), response(5000, match={"recipe": "copy"}))
    book = RecipeBook({"copy": ResourceService(service_curve(parse_case(raw), copy_op(2000)))})
    operations = tuple(replace(copy_op(2000), op_id=f"copy{i}", stream=f"copy{i}")
                       for i in range(3))
    result = graph_costs(raw, operations, recipes=book, input_ready_ms=7, isolated=False)
    assert result["complete_ms"] == 14
    assert result["execution"]["serviced"]["memory@r0:s0"] == pytest.approx(6000, rel=1e-12)
    clean(result)

def test_coverage_and_overlap_violations_are_rejected():
    support = {"q": {"min": 1, "max": 256}}
    rejects("interior_gap", rule("low", 1, 128, support=support),
            rule("high", 130, 256, support=support))
    rejects("interior_gap", rule("low", 1, 128, support=support),
            rule("high", 129, 256, support=support))
    rejects("interior_gap", rect("bottom", 1, 200, 1, 50), rect("top", 1, 200, 150, 200),
            rect("left", 1, 50, 50, 150), rect("right", 150, 200, 50, 150))
    rejects("interior_gap", rect("west", 1, 100, 1, 200), rect("east", 100, 200, 1, 100))
    rejects("interior_overlap", rect("wide", 1, 150, 1, 200), rect("narrow", 100, 200, 1, 200))

def test_missing_coordinate_in_a_contract_is_an_explicit_error():
    broken = rule("missing", 1, 200, response=affine(),
                  coordinates={"q": {"path": "metadata.absent_q"}})
    with pytest.raises(ModelingError, match="requires_declared_coordinate:missing"):
        graph_costs(engine((broken,)), (copy_op(4000),), isolated=False)

def test_ungrouped_rule_overlapping_a_covered_domain_is_rejected():
    covered = rule("covered", 1, 200)
    legacy = {"id": "legacy", "match": {"recipe": "copy"},
              "coordinates": {"q": {"path": "semantic_shape.0"}},
              "domain": {"q": {"min": 1, "max": 200}},
              "response": {"kind": "affine", "fixed_us": 8000}}
    with pytest.raises(ModelingError, match="requires_coverage_contract"):
        compiled(covered, legacy)

def test_shared_face_limits_and_knot_coverage_are_enforced():
    assert compiled(rule("low", 1, 100, response=affine(8000.0)), rule("high", 100, 200, response=affine(8000.0)))[0]["members"] == [ "low", "high"]
    rejects("seam_limit_mismatch", rule("low", 1, 100, response=affine(8000.0)),
            rule("high", 100, 200, response=affine(9000.0)))
    rejects("interior_gap", rule("partial", 1, 200,
                                 response=curve(knots=((50, 4000.0), (150, 6000.0)))))
    rejects("rule_outside_support", rule("detached", 1, 200,
                                         response=curve(knots=((300, 4000.0),
                                                               (400, 6000.0)))))
    def scaled(fixed):
        return {"kind": "affine", "fixed_us": fixed, "terms": [
            {"coordinate": "q", "us_per_unit": 1.0,
             "scale_with": {"resource": "hbm", "reference_rate": 1e6}}]}
    rejects("provable_equivalence", rule("low", 1, 100, response=scaled(8000.0)),
            rule("high", 100, 200, response=scaled(9000.0)))
    compiled(rule("low", 1, 100, response=scaled(8000.0)),
             rule("high", 100, 200, response=scaled(8000.0)))

def test_exclusive_support_bounds_and_validation_budget_fail_explicitly():
    exclusive_max = {"q": {"min": 1, "max": 200, "max_inclusive": False}}
    compiled(rule("low", 1, 100, support=exclusive_max),
             rule("high", 100, 200, support=exclusive_max))
    rejects("interior_gap", rule("low", 1, 100, support=exclusive_max),
            rule("high", 100, 199, support=exclusive_max))
    exclusive_min = {"q": {"min": 1, "max": 200, "min_inclusive": False}}
    compiled(rule("low", 1, 100, support=exclusive_min),
             rule("high", 100, 200, support=exclusive_min))
    rejects("interior_gap", rule("low", 2, 100, support=exclusive_min),
            rule("high", 100, 200, support=exclusive_min))
    coordinates = {f"x{index}": {"path": f"semantic_shape.{index}"} for index in range(4)}
    support = {name: {"min": 1, "max": 200} for name in coordinates}
    declarations = [{"id": f"box{index}", "match": {"recipe": "copy"},
                     "coordinates": coordinates,
                     "domain": {name: {"min": 1 + index * 20, "max": 20 + index * 20}
                                for name in coordinates},
                     "response": {"kind": "affine", "fixed_us": 8000},
                     "continuity": {"group": "many", "support": support,
                                    "outside": "analytic"}}
                    for index in range(8)]
    rejects("validation_budget_exceeded", *declarations)

def test_response_declarations_reject_booleans_nonfinite_and_invalid_rates():
    coordinates = {"q": {"path": "semantic_shape.0"}}
    invalid = (
        {"kind": "affine", "fixed_us": True},
        {"kind": "piecewise_linear", "coordinate": "q",
         "knots": [[1, float("nan")], [2, 3]]},
        {"kind": "multilinear", "grid": [{"coordinate": "q", "knots": [1, 2]}],
         "values_us": [1, float("inf")]},
        {"kind": "piecewise_linear", "coordinate": "q", "knots": [[1, 1], [1, 2]]},
        {"kind": "multilinear", "grid": [{"coordinate": "q", "knots": [1, 2]}],
         "values_us": [1]},
        {"kind": "affine", "terms": [{"coordinate": "q", "us_per_unit": 1,
                                      "scale_with": {"resource": "hbm",
                                                     "reference_rate": 0}}]},
        {"kind": "roofline", "services": [{"coordinate": "q", "resource": "cube",
                                           "dtype": "bf16", "utilization": True}]})
    for declaration in invalid:
        with pytest.raises(ModelingError):
            ResponseFunction.from_mapping(declaration, coordinates)

def test_response_evaluation_rejects_invalid_values_rates_and_extrapolation():
    coordinates = {"q": {"path": "semantic_shape.0"}}
    function = ResponseFunction.from_mapping({"kind": "piecewise_linear",
        "coordinate": "q", "knots": [[1, 10], [2, 20]]}, coordinates)
    decreasing = ResponseFunction.from_mapping({"kind": "piecewise_linear",
        "coordinate": "q", "outside": "linear", "knots": [[1, 100], [2, 1]]}, coordinates)
    roofline = ResponseFunction.from_mapping({"kind": "roofline", "services": [
        {"coordinate": "q", "resource": "cube", "dtype": "bf16"}]}, coordinates)
    scaled = ResponseFunction.from_mapping({"kind": "affine", "fixed_us": 1,
        "terms": [{"coordinate": "q", "us_per_unit": 1,
                   "scale_with": {"resource": "hbm", "reference_rate": 1e6}}]}, coordinates)
    calls = (lambda: function.evaluate({"q": True}), lambda: function.evaluate({}),
             lambda: nonnegative(float("nan")), lambda: nonnegative(float("inf")),
             lambda: positive(0), lambda: scaled_coordinate(1e308, 10),
             lambda: decreasing.evaluate({"q": 1000}),
             lambda: roofline.evaluate({"q": 1}, rate_for=lambda term: 0),
             lambda: roofline.evaluate({"q": 1}, rate_for=lambda term: float("nan")),
             lambda: scaled.evaluate({"q": 1}))
    for call in calls:
        with pytest.raises(ModelingError):
            call()

def test_ambiguous_exterior_groups_are_rejected_regardless_of_declaration_order():
    left = rule("left", 1, 100, group="left-group",
                support={"q": {"min": 1, "max": 100}},
                outside="linear",
                response=curve(outside="linear", knots=((1, 8000.0), (100, 10000.0))))
    right = rule("right", 200, 300, group="right-group",
                 support={"q": {"min": 200, "max": 300}},
                 outside="linear",
                 response=curve(outside="linear", knots=((200, 5000.0), (300, 8000.0))))
    for declarations in ((left, right), (right, left)):
        compiled(*declarations)
        with pytest.raises(ModelingError, match="ambiguous_exterior_groups"):
            graph_costs(engine(declarations), (replace(copy_op(4000), semantic_shape=(150.0,)),),
                        isolated=False)

def test_disjoint_guards_and_large_exterior_distances_are_accepted_and_selected():
    coords = {"q": {"path": "semantic_shape.0"}, "kv_len": {"path": "semantic_shape.1"}}
    covered = rule("covered", 1, 200, coordinates=coords, response=curve(),
                   extra_domain={"kv_len": {"min": 1000, "max": 2000}})
    unrelated = rule("unrelated", 1, 200, coordinates=coords,
                     extra_domain={"kv_len": {"min": 3000, "max": 4000}})
    unrelated.pop("continuity", None)
    assert len(compiled(covered, unrelated)) == 1
    second_group = rule("second", 1, 200, group="second-group", coordinates=coords,
                        response=curve(), extra_domain={"kv_len": {"min": 3000, "max": 4000}})
    outside = graph_costs(engine((covered, second_group)),
                          (replace(copy_op(4000), semantic_shape=(50.0, 2500.0)),),
                          isolated=False)
    assert outside["duration_ms"] == 4
    hbm_coords = {"q": {"path": "work.hbm"}}
    small = rule("small", 1, 100, coordinates=hbm_coords, outside="linear",
                 response=curve(outside="linear", knots=((1, 8000.0), (100, 9000.0))))
    large = rule("large", 100, 200, coordinates=hbm_coords, outside="linear",
                 response=curve(outside="linear", knots=((100, 9000.0), (200, 11000.0))))
    extrapolated = graph_costs(engine((small, large)), (copy_op(1_000_000_000_000),),
                               isolated=False)
    assert extrapolated["parameters"]["calibration"] == "observable_kernel_regimes"
    assert "large" not in extrapolated["parameters"]["regime_selection"]["unselected"]
    low_guard = rule("low-guard", 1, 100, coordinates=coords, response=curve(),
                     extra_domain={"kv_len": {"min": 1000, "max": 2000}})
    high_guard = rule("high-guard", 100, 200, coordinates=coords, response=curve(),
                      extra_domain={"kv_len": {"min": 3000, "max": 4000}})
    rejects("members_differ:copy-domain:guards", low_guard, high_guard)
    open_low = rule("open-low", 1, 100, coordinates=coords,
                    response=curve(knots=((1, 8000.0), (100, 9000.0))),
                    extra_domain={"kv_len": {"min": 8192}})
    open_high = rule("open-high", 100, 200, coordinates=coords,
                     response=curve(knots=((100, 9000.0), (200, 10000.0))),
                     extra_domain={"kv_len": {"min": 8192}})
    assert compiled(open_low, open_high)[0]["guards"]["kv_len"] == {
        "min": 8192.0, "max": None, "min_inclusive": True, "max_inclusive": True
    }
    base_joint = joint_case()
    execution = base_joint["runtime"]["kernels"]["service_regimes"][0]["execution"]
    base_joint["runtime"]["kernels"]["service_regimes"] = [
        rule("low", 1, 100, response=curve(), execution=execution),
        rule("high", 100, 200, response=curve(), execution=execution)]
    forward_ops = tuple(replace(copy_op(4000), op_id=str(i), semantic_shape=(s,))
                        for i, s in enumerate((50, 150)))
    reverse_ops = tuple(replace(copy_op(4000), op_id=str(i), semantic_shape=(s,))
                        for i, s in enumerate((150, 50)))
    forward = graph_costs(base_joint, forward_ops,
                          isolated=False)["parameters"]["joint_native_services"]
    reverse = graph_costs(base_joint, reverse_ops,
                          isolated=False)["parameters"]["joint_native_services"]
    assert forward == reverse
    assert forward[0]["rule_ids"] == ["high", "low"]
    assert forward[0]["rule_counts"] == {"high": 1, "low": 1}

@pytest.mark.parametrize("field,value", [("capacity_bytes_per_ms", 5e5), ("latency_us", 10)])
def test_changed_link_exits_native_response_and_retains_exact_analytic_program(field, value):
    raw, ops = source_call()
    matched = graph_costs(raw, ops, input_ready_ms=7, isolated=False, trace_limit=10000)
    assert matched["duration_ms"] == 5
    assert matched["parameters"]["service_regimes"][0]["queries"] == {"observable_response": 1}
    changed = deepcopy(raw)
    changed["machine"]["links"][0][field] = value
    assert sum(event["task"].endswith(":minimum") for event in matched["events"]) == 1
    observed = graph_costs(changed, ops, input_ready_ms=7, isolated=False, trace_limit=10000)
    del changed["runtime"]["kernels"]["service_regimes"]
    analytic = graph_costs(changed, ops, input_ready_ms=7, isolated=False, trace_limit=10000)
    assert observed["parameters"]["service_regimes"][0]["queries"] == {"physical_identity_mismatch": 1}
    assert observed["duration_ms"] == analytic["duration_ms"] < matched["duration_ms"]
    assert observed["events"] == analytic["events"]
    assert observed["execution"] == analytic["execution"]
    assert observed["execution"]["serviced"] == matched["execution"]["serviced"]
    assert observed["execution"]["live_tasks"] == observed["execution"]["live_programs"] == 0

def test_observable_graph_and_serving_share_late_ready_response_and_work():
    raw = declared_rule(response_rule(match={"recipe": copy_op(4000).recipe}))
    raw["workload"]["requests"][0]["arrival_ms"] = 7
    isolated = operator_costs(raw, (copy_op(4000),), input_ready_ms=7)
    graph = graph_costs(raw, (copy_op(4000),), input_ready_ms=7)
    serving = predict(raw, lower=lower_copy)
    assert isolated["operator_sum_ms"] == graph["duration_ms"] == 9
    assert graph["complete_ms"] == serving["requests"][0]["finish_ms"] == 16
    assert {k: v for k, v in graph["execution"]["serviced"].items() if v} == {
        "memory@r0:s0": 4000}
    assert isolated["operators"][0]["service_before_input"] == {}
    assert graph["parameters"]["service_regimes"][0]["queries"] == {"observable_response": 1}
    assert "observable_kernel_regimes" in graph["parameters"]["calibration"]
    assert serving["execution"]["live_tasks"] == serving["execution"]["live_programs"] == 0
    control = graph_costs(raw, (copy_op(4000),), isolated=False)
    raw["model"]["name"] = "unseen architecture label"
    raw["runtime"]["kernels"]["service_regimes"][0]["provenance"][
        "validation_observed_us"] = 1e9
    changed = graph_costs(raw, (copy_op(4000),), isolated=False)
    assert control["duration_ms"] == changed["duration_ms"] == 9
    assert control["execution"]["serviced"] == changed["execution"]["serviced"]

def test_native_group_prices_one_response_without_stacking_it_on_phases_or_startup():
    raw = declared_rule({"id": "native", "match": {"kernel": "NativeDispatch"},
                         "response": {"fixed_us": 10000},
                         "provenance": {"boundary": "device_kernel"}})
    raw["runtime"]["kernels"]["invocations"] = [{
        "match": {"kernel": "NativeDispatch"}, "startup_ms": 3}]
    result = graph_costs(raw, fork(), input_ready_ms=11, isolated=False)
    assert result["complete_ms"] == 21
    assert result["parameters"]["service_regimes"][0]["queries"] == {"observable_response": 1}
    phases = operator_costs(raw, fork())
    assert result["execution"]["serviced"]["memory@r0:s0"] == 5000
    assert phases["operator_sum_ms"] == 8
    assert all("service_regimes" not in row["basis"] for row in phases["operators"])

def test_prepared_source_schedule_selects_group_service_and_domain_keeps_analytic_path():
    def plan(operation, geometry):
        work = Work(resources={"hbm": 2000})
        return OperatorPlan(work, lambda c: KernelProgram((Stage("read", c.bind(work)),)),
                            metadata={"group_visits": (2, 4), "branch": "small"})
    operation = replace(op("source"), recipe="group-source", source_boundary="kernel")
    recipes = RecipeBook({"group-source": PlannedRecipe(plan)})
    rule = {"id": "small-groups",
            "match": {"metadata.program_execution.branch": "small"},
            "coordinates": {"groups": {"path": "metadata.program_execution.group_visits",
                                       "reduce": "max"}},
            "domain": {"groups": {"min": 2, "max": 4}},
            "response": {"terms": [{"coordinate": "groups", "us_per_unit": 3000}]}}
    result = graph_costs(declared_rule(rule), (operation,), recipes=recipes, isolated=False)
    assert result["duration_ms"] == 12
    assert {k: v for k, v in result["execution"]["serviced"].items() if v} == { "memory@r0:s0": 2000}
    rule["domain"]["groups"]["max"] = 3
    outside = graph_costs(declared_rule(rule), (operation,), recipes=recipes, isolated=False)
    assert outside["duration_ms"] == 2
    assert outside["parameters"]["service_regimes"][0]["queries"] == {"analytic_outside_domain": 1}

@pytest.mark.parametrize("recipe", ("qkv_projection", "attention_output", "dense_down"))
def test_same_supply_parameters_follow_tp2_to_tp8_and_large_prefill_without_point_rules(recipe):
    source = geometry_case(2)
    rule = rule_for(source, lowered(source, recipe, 1), roofline())
    original = deepcopy(rule)
    shapes = {}
    for tp, rows in product((2, 8), (1, 4096)):
        raw = bind(geometry_case(tp), rule)
        operation = lowered(raw, recipe, rows)
        m, n, k = operation.local_shape
        shapes[tp, rows] = (m, n, k)
        expected = .003 + max(2 * m * n * k / 5e8,
                              2 * (m * k + k * n + m * n) / 2.5e8) * 1000
        result = graph_costs(raw, (operation,), input_ready_ms=7)
        isolated = operator_costs(raw, (operation,), input_ready_ms=7)["operators"][0]
        assert operation.flops == 2 * m * n * k
        assert operation.hbm_bytes == 2 * (m * k + k * n + m * n)
        assert result["duration_ms"] == pytest.approx(expected)
        assert isolated["duration_ms"] == pytest.approx(expected)
        assert result["complete_ms"] == pytest.approx(7 + expected)
        assert result["execution"]["work"]["flops"] == operation.flops
        assert result["execution"]["work"]["hbm_bytes"] == operation.hbm_bytes
        assert not isolated["service_before_input"]
        record = result["parameters"]["service_regimes"][0]
        assert record["response_relations"] == {"physical_supply": 1}
        assert result["parameters"]["calibration"] == "none"
        assert record["coordinate_ranges"]["m"] == {"min": rows, "max": rows}
    assert shapes[2, 1][1] * shapes[2, 1][2] == 4 * shapes[8, 1][1] * shapes[8, 1][2]
    assert rule == original

@pytest.mark.parametrize("field", ("vllm_impl", "vllm_version", "kernel_format"))
def test_empirical_grid_does_not_cross_changed_execution_identity(field):
    raw = geometry_case(2)
    op = lowered(raw, "qkv_projection", 1)
    rule = rule_for(raw, op, {"kind": "multilinear", "grid": [
        {"coordinate": "m", "knots": [1, 8192]}], "values_us": [10000, 100000]})
    rule["coordinates"] = {key: rule["coordinates"][key] for key in ("m", "n", "k")}
    rule["domain"] = {key: {"min": value, "max": value}
                      for key, value in zip(("n", "k"), op.local_shape[1:])}
    assert graph_costs(bind(raw, rule), (op,), isolated=False)["duration_ms"] == 10
    raw["runtime"][field] = "different-implementation"
    changed = graph_costs(raw, (op,), isolated=False)
    raw["runtime"]["kernels"] = {}
    analytical = graph_costs(raw, (op,), isolated=False)
    assert changed["duration_ms"] == analytical["duration_ms"]

def test_frozen_response_surface_prices_all_prefill_chunks_without_reinstallation():
    raw = geometry_case(8)
    oracle = lambda q, kv: 850.0 + 0.08 * q + 0.068 * kv
    q_knots, kv_knots = [4096, 8192], [8192, 32768, 65536]
    rule = {
        "id": "prefill-chunk-fa-surface",
        "match": {"phase": "prefill", "recipe": "flash_attention"},
        "coordinates": {"q": {"path": "semantic_shape.0"},
                        "kv_len": {"path": "semantic_shape.1"}},
        "domain": {"q": {"min": 4096, "max": 8192},
                   "kv_len": {"min": 8192, "max": 65536}},
        "continuity": {"group": "prefill-chunk-fa",
                       "support": {"q": {"min": 4096, "max": 8192},
                                   "kv_len": {"min": 8192, "max": 65536}},
                       "outside": "analytic"},
        "response": {"kind": "multilinear",
                     "grid": [{"coordinate": "q", "knots": q_knots},
                              {"coordinate": "kv_len", "knots": kv_knots}],
                     "values_us": [oracle(q, kv) for q in q_knots for kv in kv_knots],
                     "outside": "analytic"},
        "execution": {"kind": "joint_native", "resource": "matrix_service",
                      "reference_rate": 1e9, "native_kernel": "FlashAttentionScore"}}
    bind(raw, rule)
    frozen = deepcopy(raw)
    for chunk_index in range(1, 9):
        kv_len = 8192 * chunk_index
        op = lowered(raw, "flash_attention", 8192, context=kv_len - 8192)
        result = graph_costs(raw, (op,), input_ready_ms=3, isolated=False)
        assert result["duration_ms"] == pytest.approx(oracle(8192, kv_len) / 1000)
        assert raw == frozen

def test_explicit_extrapolation_is_distinct_from_analytic_outside_curve_and_domain():
    safe = graph_costs(curve_case(), (operation(5000),), isolated=False)
    assert safe["duration_ms"] == 5
    assert safe["parameters"]["service_regimes"][0]["queries"] == { "analytic_outside_knots": 1}
    assert safe["parameters"]["regime_selection"]["unselected"] == ["observed-copy"]
    extrapolated = graph_costs(curve_case(outside="linear"), (operation(5000),), isolated=False)
    assert extrapolated["duration_ms"] == 10.5
    assert extrapolated["parameters"]["service_regimes"][0]["response_relations"] == {"extrapolated": 1}
    limited = graph_costs(curve_case(outside="linear"), (operation(9000),), isolated=False)
    assert limited["duration_ms"] == 9
    assert limited["parameters"]["service_regimes"][0]["queries"] == {"analytic_outside_domain": 1}

def test_serving_context_growth_uses_unseen_coordinate_without_refitting_or_point_rules():
    raw = curve_case()
    rule = raw["runtime"]["kernels"]["service_regimes"][0]
    rule.update(coordinates={"context": {"path": "metadata.context_tokens"}},
                domain={"context": {"min": 1, "max": 3}},
                response={"kind": "piecewise_linear", "coordinate": "context",
                          "knots": [[1, 4000], [3, 9000]]})
    original = deepcopy(rule)
    raw["workload"]["requests"][0].update(output_tokens=3, arrival_ms=7)
    def growing_context(plan, model, runtime, machine):
        view = plan.requests[0]
        return IterationWork((operation(1000, context=view.kv_len),), plan, 1, 1, 1,
                             None, view.phase, "synthetic_context_growth")
    result = predict(raw, lower=growing_context)
    assert result["requests"][0]["finish_ms"] == 7 + 4 + 6.5 + 9
    record = result["parameters"]["service_regimes"][0]
    assert record["response_relations"] == {"anchor": 2, "interpolated": 1}
    assert record["coordinate_ranges"] == {"context": {"min": 1, "max": 3}}
    assert raw["runtime"]["kernels"]["service_regimes"][0] == original

def test_compiled_knot_coverage_selects_exactly_one_rule():
    low = rule("low-knots", 1, 200, response=curve(knots=((1, 8000.0), (100, 9000.0))))
    high = rule("high-knots", 1, 200, response=curve(knots=((100, 9000.0), (200, 10000.0))))
    for declarations in ((low, high), (high, low)):
        raw = engine(declarations)
        assert select(raw, (50.0,))[0] == "low-knots"
        assert select(raw, (100.0,))[0] == "high-knots"
        assert select(raw, (150.0,))[0] == "high-knots"

@pytest.mark.parametrize("model,family", [(None, "gqa_decode"), ("kimi-k3", "kda_decode")])
def test_observed_service_regime_keeps_default_attention_algorithm(model, family):
    if model:
        raw = compose_case(model, hardware="Ascend950DT_9582", tp=8, dp=1, ep=1, num_devices=8, layers=1)
    else:
        raw = case(output=1)
        raw["machine"] = single_device_machine()
    raw["runtime"]["kernels"] = { "service_regimes": [{"id": "observed", "mode": "memory", "phases": ["decode"]}]}
    effective, report = attention_defaults(parse_case(raw), "decode", tokens=1)
    decision = next(row for row in report["selections"] if row["family"] == family)
    assert decision["selection"] == "default_installed"
    if family == "gqa_decode":
        recipe = effective.runtime.kernels["recipes"]["flash_attention_decode"]
        assert recipe["kind"] == "staged_gqa_decode"
    else:
        kda = effective.runtime.kernels["lowerings"]["kda"]
        assert (kda if isinstance(kda, str) else kda["implementation"]).startswith(
            "vllm_ascend_recurrent_kda_")

def test_quantized_gqa_uses_the_declared_fattention_plan_and_preserves_override():
    raw = case(output=1)
    raw["machine"] = single_device_machine()
    parsed = parse_case(raw)
    parsed = replace(parsed, model=replace(parsed.model, quant=replace( quant_preset("w8a8_dynamic"), kv_dtype="int8")))
    effective, report = attention_defaults(parsed, "decode", tokens=1)
    decision = next(row for row in report["selections"] if row["family"] == "gqa_decode")
    assert decision["selection"] == "default_installed"
    recipe = effective.runtime.kernels["recipes"]["flash_attention_decode"]
    assert recipe["kind"] == "staged_gqa_decode"
    explicit, explicit_report = attention_defaults( replace(parsed, runtime=replace(parsed.runtime, kernels={ "recipes": {"flash_attention_decode": {"kind": "staged_gqa_decode", "kv_splits": 2}}})), "decode", tokens=1)
    explicit_decision = next(row for row in explicit_report["selections"] if row["family"] == "gqa_decode")
    assert explicit_decision["reason"] == "explicit_implementation_preserved"
    assert explicit.runtime.kernels["recipes"]["flash_attention_decode"]["kv_splits"] == 2

def test_kda_respects_raw_eager_and_explicit_analytical_policy():
    raw = compose_case("glm-5.3-flash", hardware="a5", tp=1, ep=1, num_devices=1, layers=4, raw_eager=True, prompt_tokens=8, output_tokens=2)
    graphs = target_graphs(parse_case(raw), phase="decode", tokens=1, context=0, sequences=1, scope="model")
    scans = by_prefix(graphs[0], "s0.l0.recurrent")
    assert scans and all("vllm_ascend" not in op.recipe for op in scans)
    raw["runtime"]["kernels"] = {"lowerings": {"kda": { "cases": [{"domain": {"phases": ["decode"], "min_query_tokens": {"min": 2}, "max_query_tokens": {"max": 8}}, "binding": "vllm_ascend_recurrent_kda_arch35"}], "otherwise": "analytical"}}}
    analytical = target_graphs(parse_case(raw), phase="decode", tokens=1, context=0, sequences=1, scope="model")
    native = target_graphs(parse_case(raw), phase="decode", tokens=3, context=0, sequences=1, scope="model")
    analytical_ops = by_prefix(analytical[0], "s0.l0.recurrent")
    native_ops = by_prefix(native[0], "s0.l0.recurrent")
    assert analytical_ops and all("vllm_ascend" not in op.recipe for op in analytical_ops)
    assert native_ops and all("vllm_ascend" in op.recipe for op in native_ops)

def test_bare_and_excluded_native_kda_run_the_analytical_source():
    def bare(**options):
        raw = compose_case("glm-5.3-flash", hardware="a5", tp=8, dp=1, ep=8, num_devices=8, layers=4, **options)
        return target_graphs(parse_case(raw), phase="decode", tokens=1, context=0, sequences=1, scope="model")[0]
    for options in ({"va_recommended": False}, {"exclude_optimizations": ("recurrent_decode_native_kernels",)}):
        graph = bare(**options)
        assert not [op for op in graph if op.recipe == "vllm_ascend_recurrent_kda_arch35"]
        assert len([op for op in graph if op.family == "recurrent_scan"]) == 3
